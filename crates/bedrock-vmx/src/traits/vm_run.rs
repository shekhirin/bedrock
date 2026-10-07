// SPDX-License-Identifier: GPL-2.0

//! VM run loop and GPR synchronization.

#[cfg(not(feature = "cargo"))]
use super::super::prelude::*;
#[cfg(feature = "cargo")]
use crate::prelude::*;

use super::{
    CowAllocator, InstructionCounter, IrqGuard, Kernel, Machine, Page, ReverseIrqGuard,
    VirtualMachineControlStructure, VmContext, VmRunError, VmRunner,
};

// ========== GPR Sync Methods ==========

/// Copy GPRs into `VmxContext` and set up the XSAVE area pointers.
pub fn sync_gprs_to_vmx_ctx<V, I>(state: &mut VmState<V, I>)
where
    V: VirtualMachineControlStructure,
    I: InstructionCounter,
{
    state.vmx_ctx.guest_rax = state.gprs.rax;
    state.vmx_ctx.guest_rbx = state.gprs.rbx;
    state.vmx_ctx.guest_rcx = state.gprs.rcx;
    state.vmx_ctx.guest_rdx = state.gprs.rdx;
    state.vmx_ctx.guest_rsi = state.gprs.rsi;
    state.vmx_ctx.guest_rdi = state.gprs.rdi;
    state.vmx_ctx.guest_rbp = state.gprs.rbp;
    state.vmx_ctx.guest_r8 = state.gprs.r8;
    state.vmx_ctx.guest_r9 = state.gprs.r9;
    state.vmx_ctx.guest_r10 = state.gprs.r10;
    state.vmx_ctx.guest_r11 = state.gprs.r11;
    state.vmx_ctx.guest_r12 = state.gprs.r12;
    state.vmx_ctx.guest_r13 = state.gprs.r13;
    state.vmx_ctx.guest_r14 = state.gprs.r14;
    state.vmx_ctx.guest_r15 = state.gprs.r15;
    // RSP lives in the VMCS.

    state.vmx_ctx.guest_xsave_ptr = state.guest_xsave_page.virtual_address().as_u64();
    state.vmx_ctx.host_xsave_ptr = state.host_xsave_page.virtual_address().as_u64();
    state.vmx_ctx.xcr0_mask = state.xcr0_mask;
}

/// Copy GPRs out of `VmxContext`.
pub fn sync_gprs_from_vmx_ctx<V, I>(state: &mut VmState<V, I>)
where
    V: VirtualMachineControlStructure,
    I: InstructionCounter,
{
    state.gprs.rax = state.vmx_ctx.guest_rax;
    state.gprs.rbx = state.vmx_ctx.guest_rbx;
    state.gprs.rcx = state.vmx_ctx.guest_rcx;
    state.gprs.rdx = state.vmx_ctx.guest_rdx;
    state.gprs.rsi = state.vmx_ctx.guest_rsi;
    state.gprs.rdi = state.vmx_ctx.guest_rdi;
    state.gprs.rbp = state.vmx_ctx.guest_rbp;
    state.gprs.r8 = state.vmx_ctx.guest_r8;
    state.gprs.r9 = state.vmx_ctx.guest_r9;
    state.gprs.r10 = state.vmx_ctx.guest_r10;
    state.gprs.r11 = state.vmx_ctx.guest_r11;
    state.gprs.r12 = state.vmx_ctx.guest_r12;
    state.gprs.r13 = state.vmx_ctx.guest_r13;
    state.gprs.r14 = state.vmx_ctx.guest_r14;
    state.gprs.r15 = state.vmx_ctx.guest_r15;
}

// ========== VM Run Methods ==========

/// Run the VM until an exit requiring userspace handling. Swaps the MSRs that
/// have no VMCS fields around the run loop.
///
/// # Safety
///
/// VMCS (including HOST_RIP) must be configured, interrupts in an appropriate
/// state, and preemption disabled so the thread cannot migrate.
pub unsafe fn run<Ctx, R, M, A>(
    ctx: &mut Ctx,
    runner: &mut R,
    machine: &M,
    allocator: &mut A,
) -> Result<ExitReason, VmRunError>
where
    Ctx: VmContext,
    R: VmRunner<Vmcs = Ctx::Vmcs>,
    M: Machine,
    A: CowAllocator<Ctx::CowPage>,
{
    let msr = machine.msr_access();

    // Save host KERNEL_GS_BASE (per-thread, changes between runs).
    let host_kernel_gs_base = msr.read_msr(msr::IA32_KERNEL_GS_BASE).unwrap_or(0);

    // No VMCS fields for these; SYSCALL/SYSRET/SWAPGS read hardware directly.
    ctx.state().msr_state.syscall.load(msr);

    ctx.state().vmcs.load().map_err(VmRunError::VmcsLoad)?;

    // EPT TLB entries are per-LP and survive VM entry/exit (SDM Vol 3C
    // §30.4.3.2); cross-LP propagation is software's job (§30.4.3.4). Between
    // ioctls the thread can migrate, and CoW remaps made elsewhere may leave
    // this CPU caching stale parent HPAs. EPT violations only flush entries
    // that block the access; a permitted read through a stale entry silently
    // returns parent data (§30.4.2). So INVEPT whenever the CPU changed.
    let cur_cpu = machine.kernel().current_cpu_id() as u32;
    if ctx.state().last_cpu != Some(cur_cpu) {
        let eptp = ctx.state().ept.eptp();
        <Ctx::V as Vmx>::invept_single_context(eptp).map_err(VmRunError::InveptFailed)?;
        ctx.state_mut().last_cpu = Some(cur_cpu);
    }

    // Requires the VMCS to be loaded.
    ctx.state().apply_intercept_pf();

    // Otherwise single-stepping would only start at the first exit.
    update_mtf_state(ctx).map_err(VmRunError::ExitHandler)?;

    // SAFETY: Caller guarantees VMCS is properly configured
    let result = unsafe { run_loop(ctx, runner, machine, allocator, host_kernel_gs_base) };

    let clear_result = ctx.state().vmcs.clear();

    // After VMCLEAR the next entry must be VMLAUNCH.
    ctx.state_mut().vmx_ctx.launched = 0;

    // kernel_gs_base is already saved by run_loop on every exit.
    ctx.state_mut().msr_state.syscall = SyscallMsrs::capture(msr);

    // Restore host MSRs even if VMCLEAR failed.
    ctx.state().host_state.syscall_msrs.load(msr);
    let _ = msr.write_msr(msr::IA32_KERNEL_GS_BASE, host_kernel_gs_base);

    // VMCS is in an undefined state; re-entry could crash the host.
    if let Err(e) = clear_result {
        log_err!("VMCLEAR failed - VMCS in undefined state, cannot continue\n");
        return Err(VmRunError::VmcsClear(e));
    }

    result
}

/// Inner loop, split out so `run` clears the VMCS on every exit path.
///
/// `host_kernel_gs_base` is restored after each exit so host interrupt handlers
/// in the IRQ windows see the host value.
unsafe fn run_loop<Ctx, R, M, A>(
    ctx: &mut Ctx,
    runner: &mut R,
    machine: &M,
    allocator: &mut A,
    host_kernel_gs_base: u64,
) -> Result<ExitReason, VmRunError>
where
    Ctx: VmContext,
    R: VmRunner<Vmcs = Ctx::Vmcs>,
    M: Machine,
    A: CowAllocator<Ctx::CowPage>,
{
    // vmx_run_guest loads guest XCR0 before VMRESUME; a host interrupt in that
    // window could #UD on AVX-512. Host XCR0 is restored on exit, so the
    // ReverseIrqGuard windows between exits are safe.
    let _irq_guard = IrqGuard::new(machine.kernel());

    // Host state is constant while preemption is disabled; write it once
    // (each VMWRITE traps to L0 under nested virt).
    let cr3 = machine
        .cr_access()
        .read_cr3()
        .map_err(|_| VmRunError::ReadHostCr3)?
        .bits();
    ctx.state()
        .vmcs
        .write_natural(VmcsFieldNatural::HostCr3, cr3)
        .map_err(VmRunError::WriteHostCr3)?;

    {
        let msr = machine.msr_access();
        let fs_base = msr.read_msr(msr::IA32_FS_BASE).unwrap_or(0);
        ctx.state()
            .vmcs
            .write_natural(VmcsFieldNatural::HostFsBase, fs_base)
            .map_err(VmRunError::WriteHostFsBase)?;
        let gs_base = msr.read_msr(msr::IA32_GS_BASE).unwrap_or(0);
        ctx.state()
            .vmcs
            .write_natural(VmcsFieldNatural::HostGsBase, gs_base)
            .map_err(VmRunError::WriteHostGsBase)?;
    }

    let dta = machine.descriptor_table_access();
    ctx.state()
        .vmcs
        .write_natural(VmcsFieldNatural::HostTrBase, dta.read_tr_base())
        .map_err(VmRunError::WriteHostTrBase)?;
    ctx.state()
        .vmcs
        .write_natural(VmcsFieldNatural::HostGdtrBase, dta.read_gdtr().base)
        .map_err(VmRunError::WriteHostGdtrBase)?;

    // HOST_RSP points at VmxContext.
    let host_rsp = core::ptr::from_mut(&mut ctx.state_mut().vmx_ctx) as u64;
    ctx.state()
        .vmcs
        .write_natural(VmcsFieldNatural::HostRsp, host_rsp)
        .map_err(VmRunError::WriteHostRsp)?;

    // Must run on the loop's CPU with preemption disabled.
    ctx.state_mut()
        .instruction_counter
        .prepare()
        .map_err(VmRunError::InstructionCounter)?;

    // Auto-save/load the instruction counter across exits so host-side ticks
    // (e.g. perf's NMI handler rewriting GLOBAL_CTRL via
    // `__intel_pmu_enable_all`) are wiped on the next entry.
    if let Some(entry_phys) = ctx.state().instruction_counter.msr_save_load_entry_phys() {
        let _ = ctx
            .state()
            .vmcs
            .write64(VmcsField64::VmExitMsrStoreAddr, entry_phys);
        let _ = ctx
            .state()
            .vmcs
            .write32(VmcsField32::VmExitMsrStoreCount, 1);
        let _ = ctx
            .state()
            .vmcs
            .write64(VmcsField64::VmEntryMsrLoadAddr, entry_phys);
        let _ = ctx
            .state()
            .vmcs
            .write32(VmcsField32::VmEntryMsrLoadCount, 1);
    }

    // Hardware PERF_GLOBAL_CTRL switching, constant for the loop.
    //
    // Re-add the PEBS FIXED_CTR0 bit: the IC's snapshot lacks it, so this
    // write would otherwise clobber `register_pebs_page`'s OR and PEBS would
    // never fire.
    let pebs_registered = ctx.state().pebs_state.is_some();
    if let Some((mut guest_val, host_val)) =
        ctx.state().instruction_counter.perf_global_ctrl_values()
    {
        if pebs_registered {
            guest_val |= PERF_GLOBAL_CTRL_FIXED_CTR0;
        }
        let _ = ctx
            .state()
            .vmcs
            .write64(VmcsField64::GuestIa32PerfGlobalCtrl, guest_val);
        let _ = ctx
            .state()
            .vmcs
            .write64(VmcsField64::HostIa32PerfGlobalCtrl, host_val);

        if let Ok(entry_ctrl) = ctx.state().vmcs.read32(VmcsField32::VmEntryControls) {
            let _ = ctx.state().vmcs.write32(
                VmcsField32::VmEntryControls,
                entry_ctrl | vm_entry::LOAD_IA32_PERF_GLOBAL_CTRL,
            );
        }
        if let Ok(exit_ctrl) = ctx.state().vmcs.read32(VmcsField32::PrimaryVmExitControls) {
            let _ = ctx.state().vmcs.write32(
                VmcsField32::PrimaryVmExitControls,
                exit_ctrl | vm_exit::LOAD_IA32_PERF_GLOBAL_CTRL,
            );
        }
    }

    let loop_result = loop {
        let loop_start_tsc = rdtsc();

        ctx.sync_gprs_to_vmx_ctx();

        inject_pending_interrupt(ctx).map_err(VmRunError::ExitHandler)?;

        // Interrupt preparation may stage an event in the full trace buffer.
        // Drain before entering the guest, or its next exit record is dropped.
        if ctx.state().event_buffer_full() {
            ctx.state_mut().exit_stats.total_run_cycles += rdtsc().saturating_sub(loop_start_tsc);
            break Ok(ExitReason::EventBufferFull);
        }

        let pre_entry_tsc = rdtsc();
        ctx.state_mut().exit_stats.vmentry_overhead_cycles +=
            pre_entry_tsc.saturating_sub(loop_start_tsc);

        // IA32_KERNEL_GS_BASE has no VMCS field.
        let msr = machine.msr_access();
        let _ = msr.write_msr(msr::IA32_KERNEL_GS_BASE, ctx.state().kernel_gs_base);

        // Captured before entry: the exit handler may clear `armed_action`.
        let pebs_armed_this_iter = ctx
            .state()
            .pebs_state
            .as_deref()
            .is_some_and(|p| p.armed_action.is_some());
        if pebs_armed_this_iter {
            pebs_pre_vm_entry(ctx, msr);
        }

        // Split borrow: vmx_ctx mutably, vmcs immutably.
        let state = ctx.state_mut();
        // SAFETY: Caller guarantees VMCS is properly configured and loaded,
        // interrupts are disabled, and preemption cannot migrate us.
        let run_result = unsafe { runner.run(&mut state.vmx_ctx, &state.vmcs) };

        if pebs_armed_this_iter {
            pebs_post_vm_exit(ctx, msr);
        }

        // Before any IRQ window, so host handlers neither clobber the guest
        // value nor see it.
        ctx.state_mut().kernel_gs_base = msr.read_msr(msr::IA32_KERNEL_GS_BASE).unwrap_or(0);
        let _ = msr.write_msr(msr::IA32_KERNEL_GS_BASE, host_kernel_gs_base);

        let post_exit_tsc = rdtsc();
        ctx.state_mut().exit_stats.guest_cycles += post_exit_tsc.saturating_sub(pre_entry_tsc);

        // Prior instructions must complete before RDPMC (SDM Vol 3A §10.3 fn 3).
        // LFENCE rather than CPUID, which exits to L0 under nested virt.
        // SAFETY: LFENCE is a safe ordering instruction that ensures prior
        // instructions complete locally before RDPMC reads the performance counter.
        #[cfg(not(feature = "cargo"))]
        unsafe {
            core::arch::asm!("lfence", options(preserves_flags, nostack));
        }

        // Service pending host interrupts; host XCR0 is already restored.
        let pre_irq_tsc = rdtsc();
        {
            let _irq_window = ReverseIrqGuard::new(machine.kernel());
            let count = ctx.state().instruction_counter.read();
            ctx.state_mut().last_instruction_count = count;
        }
        let post_irq_tsc = rdtsc();
        ctx.state_mut().exit_stats.irq_window_cycles += post_irq_tsc.saturating_sub(pre_irq_tsc);

        if let Err(ref e) = run_result {
            if let Ok(error) = ctx.state().vmcs.read32(VmcsField32::VmInstructionError) {
                log_err!("VM entry failed: {:?}, VM_INSTRUCTION_ERROR={}", e, error);
            } else {
                log_err!("VM entry failed: {:?}, couldn't read error field", e);
            }
            return Err(VmRunError::VmEntry(run_result.unwrap_err()));
        }

        ctx.sync_gprs_from_vmx_ctx();

        let pre_handler_tsc = rdtsc();
        let total_exit_overhead = pre_handler_tsc.saturating_sub(post_exit_tsc);
        let irq_window = post_irq_tsc.saturating_sub(pre_irq_tsc);
        ctx.state_mut().exit_stats.vmexit_overhead_cycles +=
            total_exit_overhead.saturating_sub(irq_window);

        let kernel = machine.kernel();
        match handle_exit(ctx, kernel, allocator) {
            ExitHandlerResult::Continue => {
                ctx.finalize_exit_record(kernel);

                let loop_end_tsc = rdtsc();
                ctx.state_mut().exit_stats.total_run_cycles +=
                    loop_end_tsc.saturating_sub(loop_start_tsc);

                // cond_resched() would need preemption (and could migrate the
                // loaded VMCS); return to userspace to let the scheduler run.
                if kernel.need_resched() {
                    break Ok(ExitReason::NeedResched);
                }
                continue;
            }
            ExitHandlerResult::ExitToUserspace(reason) => {
                ctx.finalize_exit_record(machine.kernel());

                let loop_end_tsc = rdtsc();
                ctx.state_mut().exit_stats.total_run_cycles +=
                    loop_end_tsc.saturating_sub(loop_start_tsc);

                break Ok(reason);
            }
            ExitHandlerResult::Error(e) => {
                let loop_end_tsc = rdtsc();
                ctx.state_mut().exit_stats.total_run_cycles +=
                    loop_end_tsc.saturating_sub(loop_start_tsc);

                break Err(VmRunError::ExitHandler(e));
            }
        }
    };

    // Must run on the same CPU as `prepare`.
    let finish_result = ctx
        .state_mut()
        .instruction_counter
        .finish()
        .map_err(VmRunError::InstructionCounter);

    // Read once here rather than per iteration (VMCS still loaded).
    ctx.state_mut().last_exit_qualification = ctx
        .state()
        .vmcs
        .read_natural(VmcsFieldNatural::ExitQualification)
        .unwrap_or(0);
    ctx.state_mut().last_guest_physical_addr = ctx
        .state()
        .vmcs
        .read64(VmcsField64::GuestPhysicalAddr)
        .unwrap_or(0);

    finish_result?;
    loop_result
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::events::{
        EventCategories, EventKind, InjectSource, EVENT_BUFFER_SIZE, EVENT_HEADER_SIZE,
    };
    use crate::test_mocks::{MockFrameAllocator, MockMachine, MockVmcs};
    use crate::tests::MockVmContext;
    use crate::traits::{VmEntryError, VmxContext};

    #[derive(Default)]
    struct CountingRunner {
        entries: usize,
    }

    impl VmRunner for CountingRunner {
        type Vmcs = MockVmcs;

        unsafe fn run(
            &mut self,
            _ctx: &mut VmxContext,
            _vmcs: &Self::Vmcs,
        ) -> Result<(), VmEntryError> {
            self.entries += 1;
            Err(VmEntryError::VmEntryFailed)
        }
    }

    #[test]
    fn timer_event_overflow_returns_before_guest_entry() {
        let mut ctx = MockVmContext::new();
        let mut buffer = std::vec![0u8; EVENT_BUFFER_SIZE];
        ctx.state_mut().set_event_buffer(buffer.as_mut_ptr());
        ctx.state_mut()
            .set_event_categories(EventCategories::SERIAL.union(EventCategories::INJECT));

        // Fill with valid records, leaving only a header's worth of room:
        // the timer's header + 16-byte payload will have to be staged.
        let payload = [0u8; 4096];
        while EVENT_BUFFER_SIZE - ctx.state().event_buffer_len()
            >= 2 * EVENT_HEADER_SIZE + payload.len()
        {
            assert!(ctx.state_mut().event_append(EventKind::Serial, &payload));
        }
        let remaining_payload =
            EVENT_BUFFER_SIZE - ctx.state().event_buffer_len() - 2 * EVENT_HEADER_SIZE;
        assert!(ctx
            .state_mut()
            .event_append(EventKind::Serial, &payload[..remaining_payload]));
        let len_before = ctx.state().event_buffer_len();
        let seq_before = ctx.state().event_seq;
        assert!(!ctx.state().event_buffer_full());

        ctx.set_emulated_tsc(100);
        ctx.state_mut().devices.apic.svr = 1 << 8;
        ctx.state_mut().devices.apic.lvt_timer = 0xEC;
        ctx.state_mut().devices.apic.timer_deadline = 100;
        ctx.set_guest_rflags(1 << 9);
        ctx.vmcs_setup()
            .set_field32(VmcsField32::IdtVectoringInfo, 0);
        ctx.vmcs_setup()
            .set_field32(VmcsField32::GuestInterruptibilityState, 0);
        ctx.vmcs_setup()
            .set_field32(VmcsField32::PrimaryProcBasedVmExecControls, 0);

        let mut runner = CountingRunner::default();
        let mut allocator = MockFrameAllocator::new();
        // SAFETY: All hardware access, IRQ control and guest entry are mocked.
        let result = unsafe { run(&mut ctx, &mut runner, &MockMachine, &mut allocator) };
        assert!(matches!(result, Ok(ExitReason::EventBufferFull)));
        assert_eq!(runner.entries, 0);
        assert!(ctx.state().event_buffer_full());
        assert_eq!(ctx.state().event_buffer_len(), len_before);
        assert_eq!(ctx.state().event_seq, seq_before);
        assert_eq!(ctx.state().devices.apic.timer_deadline, 0);
        assert_eq!(
            ctx.vmcs_setup()
                .get_field32(VmcsField32::VmEntryInterruptionInfo),
            Some((1 << 31) | 0xEC)
        );

        ctx.state_mut().event_clear();
        assert!(!ctx.state().event_buffer_full());
        assert_eq!(ctx.state().event_buffer_len(), EVENT_HEADER_SIZE + 16);
        assert_eq!(ctx.state().event_seq, seq_before + 1);
        assert_eq!(
            u64::from_le_bytes(buffer[..8].try_into().unwrap()),
            seq_before
        );
        assert_eq!(
            u16::from_le_bytes(buffer[24..26].try_into().unwrap()),
            EventKind::Inject.as_u16()
        );
        assert_eq!(buffer[EVENT_HEADER_SIZE], 0xEC);
        assert_eq!(buffer[EVENT_HEADER_SIZE + 1], InjectSource::Timer as u8);
        assert_eq!(
            u64::from_le_bytes(
                buffer[EVENT_HEADER_SIZE + 8..EVENT_HEADER_SIZE + 16]
                    .try_into()
                    .unwrap()
            ),
            100
        );

        // After the drain, entry is allowed and the timer event isn't duplicated.
        // SAFETY: All hardware access, IRQ control and guest entry are mocked.
        let result = unsafe { run(&mut ctx, &mut runner, &MockMachine, &mut allocator) };
        assert!(matches!(
            result,
            Err(VmRunError::VmEntry(VmEntryError::VmEntryFailed))
        ));
        assert_eq!(runner.entries, 1);
        assert_eq!(ctx.state().event_seq, seq_before + 1);
    }
}

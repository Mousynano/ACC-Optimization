from tqdm import tqdm
class ProgressManager:

    def __init__(
        self,
        max_slots,
        max_iter
    ):
        self.max_iter = max_iter

        self.slot_bars = []

        for i in range(max_slots):
            bar = tqdm(
                total=max_iter,
                desc=f"Slot {i+1}",
                position=i+1,
                leave=False,
                ncols=100
            )

            bar.set_postfix_str("idle")

            self.slot_bars.append(bar)

    def assign_job(
        self,
        slot_id,
        job
    ):
        bar = self.slot_bars[slot_id]

        # reset progress
        bar.n = 0
        bar.total = self.max_iter

        bar.set_postfix_str(
            f"{job.algo_name} | "
            f"{job.fun_name} | "
            f"run {job.run_id + 1} | "
            f"{job.obj_name}"
        )

        bar.refresh()

    def update_slot(
        self,
        slot_id,
        step
    ):
        bar = self.slot_bars[slot_id]

        delta = step - bar.n

        if delta > 0:
            bar.update(delta)

    def complete_slot(
        self,
        slot_id
    ):
        bar = self.slot_bars[slot_id]

        bar.n = bar.total
        bar.refresh()

    def set_idle(
        self,
        slot_id
    ):
        self.slot_bars[slot_id].set_postfix_str(
            "idle"
        )

    def close(self):
        for bar in self.slot_bars:
            bar.close()
"""Control-only vLLM worker extension; model execution remains unchanged."""

class ReservationWorkerExtension:
    def benchmark_reserve_cache(self):
        from common import reserve_worker_cache
        return reserve_worker_cache()

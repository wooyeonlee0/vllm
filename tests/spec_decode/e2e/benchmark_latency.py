"""Benchmark the latency of processing a single batch of requests."""
from .conftest import run_greedy_equality_correctness_test
from .conftest import create_llm_generator


def main():
    common_llm_kwargs = [{
        "model": "JackFram/llama-160m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,
        "tensor_parallel_size": 4,

        # Use AsyncLLM engine, so that the engine runs in its own process.
        # Otherwise, since vLLM does not follow true SPMD, the test runner
        # process will have both the engine and the rank0 worker. NCCL is not
        # cleaned up properly, and its server host thread leaks, causing the
        # second run of the test to fail with internal NCCL error.
        "use_async": True,
    }]

    per_test_common_llm_kwargs = [{}]
    baseline_llm_kwargs = [{}]
    test_llm_kwargs = [
        {
            "speculative_model": "JackFram/llama-68m",
            "num_speculative_tokens": 5,

            # Artificially limit the draft model max model len; this forces vLLM
            # to skip speculation once the sequences grow beyond 32-k tokens.
            "speculative_max_model_len": 32,

        },
    ]

    request = None
    batch_size = 8
    output_len = 64
    seed = 1

    baseline_llm_generator = create_llm_generator("baseline", request,
                                                  common_llm_kwargs,
                                per_test_common_llm_kwargs,
                                baseline_llm_kwargs, seed)


    test_llm_generator = create_llm_generator("baseline", request,
                                              common_llm_kwargs,
                                per_test_common_llm_kwargs,
                                baseline_llm_kwargs, seed)





    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)


if __name__ == '__main__':
    main()

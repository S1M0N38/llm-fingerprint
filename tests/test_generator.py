import pytest

from llm_fingerprint.io import FileIO
from llm_fingerprint.models import Prompt, Sample
from llm_fingerprint.services import GeneratorService
from tests.utils import filter_samples


@pytest.mark.llm
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language_model,samples_num,concurrent_requests",
    [
        ("test-model-1", 1, 1),
        ("test-model-1", 2, 1),
        ("test-model-1", 3, 3),
        ("test-model-2", 1, 3),
    ],
)
async def test_generator_service(
    file_io_test: FileIO,
    prompts_test: list[Prompt],
    samples_test: list[Sample],
    language_model: str,
    samples_num: int,
    concurrent_requests: int,
):
    """Test that the GeneratorService can generate and save samples."""

    await file_io_test.save_prompts([prompt.prompt for prompt in prompts_test])

    generator = GeneratorService(
        file_io=file_io_test,
        samples_num=samples_num,
        language_model=language_model,
        concurrent_requests=concurrent_requests,
        max_tokens=100,
    )

    try:
        await generator.main()

        samples = await file_io_test.load_samples()
        samples_test = filter_samples(
            samples=samples_test,
            language_model=language_model,
            samples_num=samples_num,
        )

        assert len(samples) == len(samples_test)

        prompt_to_sample = {sample.prompt_id: sample for sample in samples_test}
        for sample in samples:
            assert isinstance(sample, Sample)
            # check that right language model is used
            assert sample.model == language_model
            # check that sample.id is valid UUID
            assert (
                isinstance(sample.id, str)
                and len(sample.id) == 36
                and all(c in "0123456789abcdef-" for c in sample.id)
            )
            # check that completion is reasonable
            # (shares some words with expected completion)
            gen_text = sample.completion.lower()
            exp_text = prompt_to_sample[sample.prompt_id].completion.lower()
            assert set(gen_text.split()) & set(exp_text.split())

    finally:
        if hasattr(generator, "language_client"):
            await generator.language_client.close()

"""Utility functions for testing LLM Fingerprint."""

from collections import defaultdict

from llm_fingerprint.models import Sample


def filter_samples(
    samples: list[Sample],
    language_model: str | None = None,
    prompt_id: str | None = None,
    prompts_num: int | None = None,
    samples_num: int | None = None,
) -> list[Sample]:
    """Filter samples based on specified criteria.
    The feature sample_test contains 18 samples (3 prompts × 2 models × 3
    variations). Sometime we are not testing against all samples so we use this
    function to filter out the samples we don't need.

    Args:
        samples: List of samples to filter
        model: Optional model name to filter by
        prompt_id: Optional prompt ID to filter by
        samples_num: Optional number of samples to filter

    Returns:
        Filtered list of samples
    """
    result = samples

    if language_model is not None:
        result = [s for s in result if s.model == language_model]

    if prompt_id is not None:
        result = [s for s in result if s.prompt_id == prompt_id]

    if prompts_num is not None:
        prompt_ids = list({s.prompt_id for s in result})[:prompts_num]
        result = [s for s in result if s.prompt_id in prompt_ids]

    if samples_num is not None:
        result_dict = defaultdict(list)
        for sample in result:
            result_dict[(sample.model, sample.prompt_id)].append(sample)

        result = []
        for sample_list in result_dict.values():
            result.extend(sample_list[:samples_num])

    return result

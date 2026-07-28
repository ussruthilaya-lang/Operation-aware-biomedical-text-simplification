# Operation-Constrained Prompts
#
# Instead of generic "simplify" prompts, use operation-specific instructions
# to guide LLM toward the intended simplification strategy

from enum import Enum


class PromptType(str, Enum):
    SUBSTITUTION = "substitution"
    EXPLANATION = "explanation"
    GENERALIZATION = "generalization"


SUBSTITUTION_PROMPT = """Replace medical jargon with plain English equivalents.

Medical sentence: {sentence}

Instructions:
- Identify medical or technical terms
- Replace them with simple, everyday English words
- Keep the same meaning
- Do not add definitions or remove details

Simplified sentence:"""


EXPLANATION_PROMPT = """Add a plain-English explanation in parentheses for medical terms.

Medical sentence: {sentence}

Instructions:
- Identify medical or technical terms
- Add explanations in parentheses right after each term
- Explanations should use simple language
- Keep all original information
- Format: "term (plain explanation)"

Simplified sentence with explanations:"""


GENERALIZATION_PROMPT = """Remove technical details while keeping the main point.

Medical sentence: {sentence}

Instructions:
- Remove numerical details, dosages, specific measurements
- Remove technical jargon that's not essential to the main message
- Keep the core clinical meaning
- Use simpler structure if possible
- Keep it concise

Simplified sentence:"""


def get_prompt(prompt_type: str, sentence: str) -> str:
    """
    Get the operation-specific prompt.

    Args:
        prompt_type: 'substitution', 'explanation', or 'generalization'
        sentence: Source medical sentence

    Returns:
        Formatted prompt for LLM
    """
    prompt_type = prompt_type.lower()

    if prompt_type == PromptType.SUBSTITUTION:
        return SUBSTITUTION_PROMPT.format(sentence=sentence)

    elif prompt_type == PromptType.EXPLANATION:
        return EXPLANATION_PROMPT.format(sentence=sentence)

    elif prompt_type == PromptType.GENERALIZATION:
        return GENERALIZATION_PROMPT.format(sentence=sentence)

    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def get_all_prompts() -> dict:
    """Return all prompt templates."""
    return {
        'substitution': SUBSTITUTION_PROMPT,
        'explanation': EXPLANATION_PROMPT,
        'generalization': GENERALIZATION_PROMPT,
    }


if __name__ == "__main__":
    # Example usage
    test_sentence = "The patient presented with acute myocardial infarction requiring immediate percutaneous coronary intervention."

    print("=" * 80)
    print("SUBSTITUTION PROMPT")
    print("=" * 80)
    print(get_prompt('substitution', test_sentence))
    print()

    print("=" * 80)
    print("EXPLANATION PROMPT")
    print("=" * 80)
    print(get_prompt('explanation', test_sentence))
    print()

    print("=" * 80)
    print("GENERALIZATION PROMPT")
    print("=" * 80)
    print(get_prompt('generalization', test_sentence))
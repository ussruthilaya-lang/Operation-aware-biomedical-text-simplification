# Operation Router: Route predictions to appropriate simplification methods
#
# Based on predicted operation, route to:
# - Substitution → CHV (Common Health Vocabulary) lookup
# - Explanation → Constrained LLM prompt (add definition)
# - Generalization → Constrained LLM prompt (remove detail)

from enum import Enum


class Operation(str, Enum):
    SUBSTITUTION = "Substitution"
    EXPLANATION = "Explanation"
    GENERALIZATION = "Generalization"


class RoutingMethod(str, Enum):
    CHV_LOOKUP = "chv_lookup"
    LLM_CONSTRAINED = "llm_constrained"


def route_operation(operation: str, sentence: str) -> dict:
    """
    Route a predicted operation to the appropriate simplification method.

    Args:
        operation: Predicted operation (Substitution/Explanation/Generalization)
        sentence: Source sentence to simplify

    Returns:
        dict: {
            'operation': str,
            'method': str (CHV_LOOKUP or LLM_CONSTRAINED),
            'prompt_type': str (for LLM routing),
            'sentence': str
        }
    """
    result = {
        'operation': operation,
        'sentence': sentence,
    }

    if operation == Operation.SUBSTITUTION:
        result['method'] = RoutingMethod.CHV_LOOKUP
        result['prompt_type'] = None
        result['description'] = "Route to CHV lookup: replace jargon with plain English"

    elif operation == Operation.EXPLANATION:
        result['method'] = RoutingMethod.LLM_CONSTRAINED
        result['prompt_type'] = 'explanation'
        result['description'] = "Use constrained LLM: add definition in parentheses"

    elif operation == Operation.GENERALIZATION:
        result['method'] = RoutingMethod.LLM_CONSTRAINED
        result['prompt_type'] = 'generalization'
        result['description'] = "Use constrained LLM: remove technical details"

    else:
        raise ValueError(f"Unknown operation: {operation}")

    return result


def route_batch(predictions: list, sentences: list) -> list:
    """
    Route a batch of predictions.

    Args:
        predictions: List of operation predictions
        sentences: List of source sentences

    Returns:
        list of routing decisions
    """
    return [
        route_operation(pred, sent)
        for pred, sent in zip(predictions, sentences)
    ]


if __name__ == "__main__":
    # Example usage
    examples = [
        ("Substitution", "This patient has myocardial infarction."),
        ("Explanation", "The drug is contraindicated in renal failure."),
        ("Generalization", "A dose-dependent improvement of bladder capacity (5-fold) was observed."),
    ]

    print("=== Operation Router Examples ===\n")
    for operation, sentence in examples:
        routing = route_operation(operation, sentence)
        print(f"Operation: {routing['operation']}")
        print(f"Method: {routing['method']}")
        if routing.get('prompt_type'):
            print(f"Prompt Type: {routing['prompt_type']}")
        print(f"Description: {routing['description']}")
        print()
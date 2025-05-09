from medgpt.intent_classifier import predict_intent


def test_classifier_basic():
    label, _ = predict_intent("I want a doctor")
    assert label in ["COR", "OOC", "INC"]

def test_cor_classification():
    label, _ = predict_intent("I want to see a dermatologist in LA")
    assert label == "COR", f"Expected COR, but got {label}"

def test_ooc_classification():
    label, _ = predict_intent("I have a headache and need a prescription")
    assert label == "OOC", f"Expected OOC, but got {label}"

def test_inc_classification():
    label, _ = predict_intent("OMG you're so annoying")
    assert label == "INC", f"Expected INC, but got {label}"

def test_empty_string():
    label, _ = predict_intent("")
    assert label == "INC", f"Expected INC, but got {label} for empty string"
    
def test_confidence_above_threshold():
    label, confidence = predict_intent("I want to see a dermatologist")
    assert confidence > 30, f"Expected confidence > 30, got {confidence}"


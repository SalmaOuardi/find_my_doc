# Intent Classification Guidelines for RyteGPT

This document outlines the labeling rules and examples used to classify user queries into three distinct intent categories: `COR`, `OOC`, and `INC`. Use these guidelines to label new data or review predictions consistently.

---

## 🟢 COR (Correct / Actionable)

**Definition:**  
User queries that are respectful, medically relevant, and include enough information to take action — e.g., to search for a doctor, specialty, or service.

**Examples:**
- "I need to see a dermatologist in New York."
- "Find me a neurologist near downtown Chicago."
- "Can you recommend a cardiologist in Boston?"
- "Looking for a pediatrician around San Francisco."

**Characteristics:**
- Clear medical need
- Specific or implicit specialty/location
- Polite or neutral language

---

## 🟡 OOC (Out of Context / Incomplete)

**Definition:**  
User queries that are medically related but do not provide enough information to search effectively, or are vague symptom descriptions without a request for help.

**Examples:**
- "I have a fever and chills."
- "My back hurts a lot these days."
- "Feeling anxious all the time."
- "Pain in chest when breathing deeply."

**Characteristics:**
- Symptom descriptions without intent to find care
- Vague or incomplete queries
- Still medically relevant

---

## 🔴 INC (Inappropriate / Irrelevant)

**Definition:**  
User queries that are offensive, irrelevant, or outside the scope of the medical assistant system.

**Examples:**
- "Get me a damn doctor now!"
- "Play a song by Taylor Swift."
- "What’s the weather in Chicago?"
- "This app is garbage."

**Characteristics:**
- Offensive or disrespectful
- Completely unrelated to medicine
- Sarcasm, jokes, or trolling

---

## 🚨 Edge Case Handling

| Query | Label | Reason |
|-------|-------|--------|
| "I want help with skin issues" | OOC | Too vague, no doctor or location specified |
| "Find a neurologist near me idiot" | INC | Offensive language |
| "Looking for help with anxiety" | COR or OOC | If framed as a help request → COR; if vague → OOC |
| "I need urgent care for my child" | COR | Specific and urgent request |

---

## ✅ Labeling Checklist

- [ ] Is the query actionable? → `COR`
- [ ] Is it medical but incomplete? → `OOC`
- [ ] Is it offensive, irrelevant, or a joke? → `INC`

---

Use this guideline consistently when labeling training, validation, or synthetic examples.

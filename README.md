# Transformer-Based Semantic Criticality and Deadline Prediction for V2V Safety Messages

## Project Overview

Vehicle-to-Vehicle (V2V) communication allows vehicles to exchange safety-related
messages such as sudden braking, collision risks, road obstacles, traffic
conditions, and emergency situations.

However, not every message has the same urgency.

This project develops a prototype machine-learning pipeline that uses a
Transformer-based NLP model to understand the semantic information in a V2V
message and predict:

1. **Criticality** – How important or urgent is the message?
2. **Deadline** – How quickly should the message be processed?

---

## Problem Statement

A simple keyword-based system may not fully understand the context of a safety
message.

For example:

> "Vehicle ahead stopped suddenly 20 m ahead."

This message indicates a potentially immediate safety situation.

The proposed system therefore uses contextual text representations from a
Transformer model before making the criticality and deadline predictions.

---

## Proposed Methodology

The prototype follows this workflow:

```text
V2V Safety Message
        ↓
Text Preprocessing
        ↓
Transformer Tokenization
        ↓
DistilBERT Transformer
        ↓
768-Dimensional Embedding
        ↓
   ┌───────────────┐
   ↓               ↓
Criticality      Deadline
Prediction       Prediction
   ↓               ↓
Low/Medium/      Response
High/Critical    Time (seconds)
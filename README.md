# CSCI 5832 NLP Fall 2025 Final Project
Amanda Hernandez Sandate, Nicole Jone, Zachary Kaufman
University of Colorado Boulder

This GitHub is part of the submission requirements for the CSCI 5832 Natural Language Processing final project. It's code includes the completion requirements for the SemEval26 Task 10. The code was based on the Task 10 Starter Pack provided by SemEval, which was verbally approved by Dr. Jim Martin. 

### SemEval Task 10 Starter Pack
[Link to Starter Pack](https://github.com/hide-ous/semeval26_task10_starter_pack)

The SemEval 2026 Task 10 Starter pack contains "Scripts to facilitate participation in the 2026 Semeval Task 10: PsyCoMark -- Psycholinguistic Conspiracy Marker Extraction and Detection" which we used as the starting point for our project.

The full README for the Task 10 Starter pack can be found in this repo under the name StartP_README.md

### Progress
For both Task 1 and Task 2 we tried three different models: distilbert (default), alberta, and google electra. In both Task 1 and Task 2, roberta performed the best (although in task 1 it was very close to distilbert). After finding the best model, a grid search of hyperparameters was conducted. Since Task 2 takes only 1/5 the amount of time to train, we were able to try more hyperparameter combinations for Task 2. All of the hyperparameters we tried are below, and all combinations were attempted:

Task 1:

    learning_rate: 1e-5, 2e-5, 3e-5, 5e-5

    batch_size: 16, 32

    num_epochs: 10, 15


Task 2:

    learning_rate: 1e-5, 2e-5, 3e-5, 5e-5

    batch_size: 16, 32

    num_epochs: 10, 15

    warmup_ratio: 0.0, 0.1

    weight_decay: 0.0, 0.01, 0.1

    gradient_accumulation_steps: 1, 2


The hyperparameters that performed best for each task are below:

Task 1:

    Hyperparameters:

        Learning Rate: 5e-5

        Batch Size: 32

        Number of Epochs: 15

        Warmup Ratio: 0.1

        Weight Decay: 0.01

        Gradient Accumulation Steps: 2


    F1 Score: 0.7847246955444285

Task 2:
    Hyperparameters:

        Learning Rate: 5e-5

        Batch Size: 16

        Number of Epochs: 15


    F1 Score (Aggregate): 0.3762537272973705

    F1 Score (Macro): 0.3637177457713751
graph TD
    %% --- Styling Definitions ---
    classDef human fill:#ff9f43,stroke:#333,stroke-width:2px,color:white;
    classDef logic fill:#54a0ff,stroke:#333,stroke-width:2px,color:white;
    classDef ai fill:#5f27cd,stroke:#333,stroke-width:2px,color:white;
    classDef math fill:#eb4d4b,stroke:#333,stroke-width:2px,color:white;
    classDef startend fill:#1dd1a1,stroke:#333,stroke-width:2px,color:white;

    %% --- Initialization ---
    Start((START)):::startend --> Init[Init Random PID & Weights]:::logic
    Init --> SetParams[<b>Set Search Params</b><br/>Mutation Rate = 0.5<br/>Exploration = Medium]:::math
    
    %% --- The Optimization Loop ---
    subgraph Optimization_Phase [Phase 2: The Search]
        direction TB
        SetParams --> RunDE[<b>Run DE</b><br/>Minimize Cost J]:::ai
        SetParams --> RunBO[<b>Run BO</b><br/>Minimize Cost J]:::ai
        
        RunDE --> CandA[Get Candidate A]:::logic
        RunBO --> CandB[Get Candidate B]:::logic
    end

    %% --- Simulation ---
    CandA & CandB --> Sim[<b>Simulate Both</b><br/>Calculate Metrics]:::logic
    Sim --> Display[<b>Display Plot</b><br/>Show Curve A vs Curve B]:::human

    %% --- Human Interaction (4-Way Choice) ---
    Display --> User{<b>HUMAN CHOICE</b><br/>What do you think?}:::human

    %% --- Branch 1 & 2: Preference Learning ---
    User -->|Prefer A| CalcDelta1[Calc Gap: B - A]:::math
    User -->|Prefer B| CalcDelta2[Calc Gap: A - B]:::math
    
    CalcDelta1 --> UpdateW1[<b>Update Weights</b><br/>Shift towards A features]:::math
    CalcDelta2 --> UpdateW2[<b>Update Weights</b><br/>Shift towards B features]:::math

    %% --- Branch 3: Tie (Consolidate) ---
    User -->|Tie: Both Good| Refine[<b>Mode: Refine</b><br/>Keep Weights Same<br/>Decrease Mutation Rate]:::logic

    %% --- Branch 4: Reject (Explore) ---
    User -->|Reject: Both Bad| Explore[<b>Mode: Explore</b><br/>Keep Weights Same<br/>Increase Mutation Rate]:::logic

    %% --- Feedback Loop ---
    UpdateW1 & UpdateW2 --> Norm[Normalize Weights]:::math
    
    Norm --> ResetParams[Reset Mutation to Normal]:::math
    
    ResetParams & Refine & Explore --> Log[Log Decision]:::logic
    
    Log --> CheckDone{Satisfied?}:::logic
    CheckDone -- No --> SetParams
    CheckDone -- Yes --> End((END)):::startend
graph TD
    subgraph Legend [Diagram Key]
        direction LR
        rect1[ ] -- PyTorch Module/Layer --> rect2(( )) -- Data/Tensor --> rect3{ } -- Operation/Decision
    end

    subgraph Input
        A_INPUT((State [20]))
    end
    
    subgraph "Shared Backbone"
        B_SHARED_NET[shared_net<br>nn.Sequential]
        A_INPUT --> B_SHARED_NET
        C_SHARED_FEATURES((Shared Features [256]))
        B_SHARED_NET --> C_SHARED_FEATURES
    end

    subgraph "Critic Network (Value Path)"
        F_CRITIC_HEAD[critic_head<br>nn.Linear]
        C_SHARED_FEATURES --> F_CRITIC_HEAD
        G_VALUE((State Value V [1]))
        F_CRITIC_HEAD --> G_VALUE
    end

    subgraph "Actor Network (Policy Path)"
        subgraph "Phase 1: Generate A1 (Autoregressive)"
            direction TB
            
            H_LOOP_CONTAINER["Autoregressive Loop<br><i>(Runs 20 times, i = 0 to 19)</i>"]
            
            subgraph H_LOOP_CONTAINER
                direction LR
                I_RNN_INPUT((RNN Input<br>[256+1]))
                J_RNN[rnn<br>nn.GRUCell]
                C_SHARED_FEATURES --> I_RNN_INPUT
                
                K_PREV_ACTION((Prev Action A1[i-1])) -- Concatenate --> I_RNN_INPUT
                I_RNN_INPUT --> J_RNN
                L_HIDDEN_STATE((Hidden State h_t))
                J_RNN --> L_HIDDEN_STATE
                
                M_A1_HEAD[action1_head<br>nn.Linear]
                L_HIDDEN_STATE --> M_A1_HEAD
                N_LOGITS((Logits for A1[i]<br>[0, 1]))
                M_A1_HEAD --> N_LOGITS
                
                O_MASKING{Dynamic Masking<br>Is ones_count < 5?}
                N_LOGITS --> O_MASKING
                P_SAMPLED_A1((Sampled A1[i]))
                O_MASKING -- Masked Logits --> P_SAMPLED_A1
                
                P_SAMPLED_A1 -- Update for next loop --> K_PREV_ACTION
                
                subgraph " "
                  direction LR
                  style Z fill:none,stroke:none
                  Z[ ]
                end
            end
        end

        subgraph "Phase 2: Generate A2 (Dependent)"
            direction TB
            L_HIDDEN_STATE -- "Final Hidden State" --> Q_A2_HEAD[action2_head<br>nn.Linear]
            Q_A2_HEAD --> R_CONCENTRATION_LOGITS((Dirichlet Conc. Logits [20]))
            R_CONCENTRATION_LOGITS --> S_SOFTPLUS[Softplus + 1]
            T_MASKED_CONC((Masked Conc. Params [20]))
            S_SOFTPLUS --> T_MASKED_CONC
            
            V_A1_FINAL((Final A1 Vector [20]))
            H_LOOP_CONTAINER -- "Collect all A1[i]" --> V_A1_FINAL
            V_A1_FINAL -- "Mask with A1" --> T_MASKED_CONC
            
            W_A2_FINAL((Final A2 Vector [20]))
            T_MASKED_CONC -- "Dirichlet Sampling" --> W_A2_FINAL
        end
    end
    
    subgraph "Final Outputs"
        X_ACTION((Action<br>(A1, A2)))
        Y_LOGPROB((Total Log Prob<br>logp(A1) + logp(A2)))
        
        W_A2_FINAL --> X_ACTION
        V_A1_FINAL --> X_ACTION
        
        H_LOOP_CONTAINER -- "Sum LogProbs" --> Y_LOGPROB
        W_A2_FINAL -- "logp(A2)" --> Y_LOGPROB
        G_VALUE --> Z_OUTPUTS(( ))
        X_ACTION --> Z_OUTPUTS
        Y_LOGPROB --> Z_OUTPUTS
        
    end

    style Legend fill:#f9f,stroke:#333,stroke-width:2px
    style H_LOOP_CONTAINER fill:#lightgrey,stroke:#333,stroke-width:2px

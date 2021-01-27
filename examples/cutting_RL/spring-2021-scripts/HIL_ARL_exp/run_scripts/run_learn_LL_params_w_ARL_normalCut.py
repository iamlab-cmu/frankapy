'''
Workflow:
- initialize GP reward model
- initialize policy
- for epoch in num_epochs:
    -for sample in num_samples:
        - sample policy params
        - get/save features from sample 
        - query human for reward (save as expert reward)
        - get/save task success metrics like before
        - get reward from GP reward model based on current reward model (GP model not trained yet if epoch =0, so need to first train)
        - save expected GP model reward mean and var to buffers and training data buffer
    - if epoch == 0:
        - initialize GP reward model w/ training data from samples in 1st epoch 
    - elif epoch!=0:
        - update policy w/ REPS
        - compute EPD for each sample (using kl_div) and get samples_to_query and queried_outcomes
        - query expert for rewards if samples_to_query!=[]
        - save all queried outcomes and queried rewards in buffer to send to GPytorch model as training data everytime it get updated
        - Add samples to query to running list of queried_samples, Keep track of number of queried samples 
        - Update Reward model GP if there are any outcomes to query
'''
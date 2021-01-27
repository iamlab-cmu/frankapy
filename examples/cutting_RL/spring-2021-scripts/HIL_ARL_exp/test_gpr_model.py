if __name__ == "__main__":
    # define train_x and train_y data
    train_x = np.random.normal(0, 1, 5)    
    train_y = get_y_from_x(train_x)

    train_x = torch.from_numpy(train_x)
    train_x = train_x.float()
    train_y = torch.from_numpy(train_y)
    train_y = train_y.float()

    # define test_x data 
    test_x = np.linspace(-5,5,500)
    test_x = torch.from_numpy(test_x)
    test_x = test_x.float()
    #import pdb; pdb.set_trace()

    # define likelihood function and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # train_x is inputted voxel data repr (?), train_y is human queried rewards (?)
    model = GPRegressionModel(train_x, train_y, likelihood) 

    # train the model 
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    # optimizer = torch.optim.Adam([
    #     {'params': model.feature_extractor.parameters()},
    #     {'params': model.covar_module.parameters()},
    #     {'params': model.mean_module.parameters()},
    #     {'params': model.likelihood.parameters()},
    # ], lr=0.01)

    # # "Loss" for GPs - the marginal log likelihood
    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # train model
    train(num_epochs = 200)
    import pdb; pdb.set_trace()   
    # evaluate model    
    #import pdb; pdb.set_trace()

    #evaluate_model(test_x)
    model.eval()
    likelihood.eval()
    print('evaluating model')
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False):  
        #import pdb; pdb.set_trace()      
        preds = model(test_x)
        observed_pred = likelihood(model(test_x))
        #import pdb; pdb.set_trace()

    print('pred mean', preds.mean)
    print('pred var', preds.variance)
    plt.plot(test_x, preds.mean)
    plt.xlabel('input')
    plt.ylabel('output from GP model')
    plt.show()

    with torch.no_grad(): 
        #Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.set_xlabel('input')
        ax.set_ylabel('output from GP model')
    plt.show()
    import pdb; pdb.set_trace()
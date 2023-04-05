import hypergrad as hg

def set_forward(self,x):
        
    def loss_train_selfused(params, hparams):
        reg_loss=self.meta_biased_reg(hparams,params)
        scores = fmodel(x_a_i,params=params)
        scores_robust = fmodel(adv_input_a,params=params)
        set_loss = self.loss_fn( scores, y_a_i)
        loss_robust = self.loss_fn(scores_robust, y_a_i)
        loss_constraint=F.softplus(loss_robust-set_loss*1.2,1.0)

        return set_loss, reg_loss,loss_constraint

    def loss_train_call(params, hparams):
        reg_loss=self.meta_biased_reg(hparams,params[:-1])
        scores = fmodel(x_a_i,params=params[:-1])
        scores_robust = fmodel(adv_input_a,params=params[:-1])
        set_loss = self.loss_fn( scores, y_a_i)
        loss_robust = self.loss_fn(scores_robust, y_a_i)
        loss_constraint=F.softplus(loss_robust-set_loss*1.2,1.0)*params[-1]

        return set_loss+reg_loss+loss_constraint
    
    def loss_train_call_unconstrained(params, hparams):
        reg_loss=self.meta_biased_reg(hparams,params)
        scores = fmodel(x_a_i,params=params)
        scores_robust = fmodel(adv_input_a,params=params)
        set_loss = self.loss_fn( scores, y_a_i)
        loss_robust = self.loss_fn(scores_robust, y_a_i)

        return set_loss+reg_loss
    
    def loss_val_call(params, hparams):
        scores=fmodel(x_b_i,params=params[:-1])
        set_loss = self.loss_fn( scores, y_b_i)
        return set_loss 
    
    def loss_val_call_unconstrained(params, hparams):
        scores=fmodel(x_b_i,params=params)
        set_loss = self.loss_fn( scores, y_b_i)
        return set_loss 
    
    def inner_loop_my(params,hparams, n_steps=self.task_update_num, create_graph=False):
        a=np.ones(shape=(1))
        multiple=torch.tensor(a, dtype=torch.float32).requires_grad_(False).cuda()
        optimizer0 = torch.optim.Adam(params,lr=self.train_lr,weight_decay=0.0)
        for i in range(n_steps):
            loss1,reg_loss,loss2=loss_train_selfused(params,hparams)
            loss_train1=loss1+reg_loss+multiple*loss2
            optimizer0.zero_grad()
            loss_train1.backward(retain_graph=True)
            optimizer0.step()

            gradient_lambada=loss2.item()-0.693 #ln 2
            if gradient_lambada>0.01:
                gradient_lambada=0.01
            elif gradient_lambada<-0.01:
                gradient_lambada=-0.01
            gradient_lambada=20.0*gradient_lambada
            multiple+=gradient_lambada
            if multiple.data<0.0: 
                multiple.data=torch.tensor(np.zeros(shape=(1)), dtype=torch.float32).requires_grad_(False).cuda()

        return [par.detach().clone().requires_grad_(False) for par in params],multiple
    fast_temp = [para.detach().clone().requires_grad_() for para in list(self.parameters())]
    
    meta_parameters_self = [para.detach().clone().requires_grad_(False) for para in list(self.parameters())]
    fast_parameters,multiple=inner_loop_my(fast_temp,meta_parameters_self)
    
    scores = fmodel(x_b_i,params=fast_parameters)
    loss = self.loss_fn(scores, y_b_i)
    scores_robust = fmodel(adv_input_b,params=fast_parameters) 
    loss_robust = self.loss_fn(scores_robust, y_b_i)
    
    return scores,scores_robust,loss,loss_robust,loss_train_call, loss_val_call,loss_train_call_unconstrained,loss_val_call_unconstrained, fast_parameters,multiple     

def compute_hyper_gradient(self, x, require_rob = True):
    scores,scores_robust,loss,loss_robust,loss_train_call, loss_val_call,loss_train_call_unconstrained,loss_val_call_unconstrained, fast_parameters,multiple = self.set_forward(x)

    if multiple.data>0.01:
        cg_fp_map = hg.GradientDescent(loss_f=loss_train_call, step_size=1.)  
        new_parameter=fast_parameters+[multiple]
        hg.CG(new_parameter, list(self.parameters()), K=5, fp_map=cg_fp_map, outer_loss=loss_val_call) 
    else:
        cg_fp_map = hg.GradientDescent(loss_f=loss_train_call_unconstrained, step_size=1.)  
        hg.CG(fast_parameters, list(self.parameters()), K=5, fp_map=cg_fp_map, outer_loss=loss_val_call_unconstrained) 

    return loss, loss_robust


# compute_hyper_gradient is to update the grad for the parameter self.parameters()
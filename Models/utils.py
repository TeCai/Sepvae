import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.distributions as torchd
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, test_loader, training_step, report_step, save_path, num_sample = 1,save_name = 'sepvae.pth', device = torch.device('cpu'),logger=None, report_train = 10):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.training_step = training_step
        self.report_step = report_step
        self.logger = logger
        self.step = 0
        self.save_path = save_path
        self.device = device
        self.report_train = report_train
        self.save_name = save_name
        self.num_sample = num_sample

        self.loss_list = []
        self.evaloss_list = []

        self.klfine_list = []
        self.klcoarse_list = []
        self.L = []
        self.evaklfine_list = []
        self.evaklcoarse_list = []
        self.evaL = []



    def evaluate(self):
        klf = []
        klc = []
        L = []
        self.model.eval()
        with torch.no_grad():
            for X in self.test_loader:
                X = X.to(self.device)
                fine_pos_mean, fine_pos_var, coarse_pos_mean, coarse_pos_var, fine_sample, coarse_sample, output = self.model(X)
                f,c,l = self.model.elbo(fine_pos_mean, fine_pos_var, coarse_pos_mean, coarse_pos_var, output, X)
                klf.append(f.item())
                klc.append(c.item())
                L.append(l.item())
        klfine = np.mean(np.array(klf))
        klcoarse = np.mean(np.array(klc))
        negalike = np.mean(np.array(L))

        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(X[0].cpu().detach().permute(1,2,0).numpy())
        ax[1].imshow(output[0].cpu().detach().permute(1,2,0).numpy())
        plt.show()

        del klf,klc,L
        return klfine,klcoarse,negalike

    def save(self, path = './'):

        all_state_dict = {
        'model_sd' : self.model.state_dict(),
        'optimizer_sd' : self.optimizer.state_dict(),
        'sche_sd' : self.scheduler.state_dict(),
        'step' : self.step,
        'loss_list' : self.loss_list,
        'evaloss_list' : self.evaloss_list
        }

        torch.save(all_state_dict, path/self.save_name)





    def train(self):
        best_eva_loss = np.inf
        self.model.to(self.device)
        while True:
            for X in self.train_loader:
                X = X.to(self.device)
                self.model.train()
                self.optimizer.zero_grad()
                fine_pos_mean, fine_pos_var, coarse_pos_mean, coarse_pos_var, fine_sample, coarse_sample, output = self.model(X,self.num_sample)
                klfine,klcoarse,negalikeli = self.model.elbo(fine_pos_mean, fine_pos_var, coarse_pos_mean, coarse_pos_var, output, X)
                loss = klfine + klcoarse + negalikeli
                self.loss_list.append(loss.item())
                self.klfine_list.append(klfine.item())
                self.klcoarse_list.append(klcoarse.item())
                self.L.append(negalikeli.item())
                if self.step % self.report_train == 0:
                    lo = np.mean(self.loss_list[-self.report_train:])
                    fo = np.mean(self.klfine_list[-self.report_train:])
                    co = np.mean(self.klcoarse_list[-self.report_train:])
                    Lo = np.mean(self.L[-self.report_train:])
                    if self.logger:
                        self.logger.info(f"step:{self.step}, loss:{lo:.6f}, finekl:{fo:.6f},coarsekl:{co:.6f},L:{Lo:.6f}")
                    else:
                        print(f"step:{self.step}, loss:{lo:.6f}, finekl:{fo:.6f},coarsekl:{co:.6f},L:{Lo:.6f}")

                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.step += 1

        # valid
                if self.step % self.report_step == 0:
                    klfine,klcoarse,negalikeli = self.evaluate()
                    evaloss = klfine + klcoarse + negalikeli
                    self.evaloss_list.append(evaloss.item())
                    self.evaklfine_list.append(klfine.item())
                    self.evaklcoarse_list.append(klcoarse.item())
                    self.evaL.append(negalikeli.item())
                    if self.logger:
                        self.logger.info(f"step:{self.step}, eval_loss:{evaloss.item():.6f}, finekl:{klfine.item():.6f},coarsekl:{klcoarse.item():.6f},L:{negalikeli.item():.6f}")
                    else:
                        print(f"step:{self.step}, eval_loss:{evaloss.item():.6f}, finekl:{klfine.item():.6f},coarsekl:{klcoarse.item():.6f},L:{negalikeli.item():.6f}")

                    # saving the best
                    if evaloss.item() < best_eva_loss:
                        best_eva_loss = evaloss.item()
                        self.save(self.save_path)

                if self.step >= self.training_step:
                    return None



class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):

    has_rsample = True

    def __init__(self, logits=None, probs=None):
        super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(torch.argmax(super().logits, dim=-1), super().logits.shape[-1])
        return _mode.detach() + super().probs - super().probs.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError('need to check')
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class STBernoulli(torchd.bernoulli.Bernoulli):

    has_rsample = True

    def __init__(self, logits=None, probs=None):
        super().__init__(logits=logits, probs=probs)

    def mode(self):
        # _mode = F.one_hot(torch.argmax(super().logits, dim=-1), super().logits.shape[-1])
        _mode = torch.where(super().probs > 0.5, torch.ones_like(super().probs), torch.zeros_like(super().probs))
        return _mode.detach() + super().probs - super().probs.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError('need to check')
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample
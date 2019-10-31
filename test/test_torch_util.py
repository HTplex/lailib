import torch
import torch.nn as nn
from torch.nn import init
from os import listdir
from lailib.torch.parameter_store import save_network, load_last_checkpoint, load_network
import pytest


# TODO add test for optimizer
# TODO test reset parameter

@pytest.fixture(scope='function')
def model_and_meta():
    model_name = 'Dummy'
    global_step = 0
    saved_model = DummyTorchModule()
    return [model_name, global_step, saved_model]


class DummyTorchModule(nn.Module):
    def __init__(self):
        super(DummyTorchModule, self).__init__()
        self.test_weight = torch.nn.Parameter(init.orthogonal_(torch.Tensor(5, 5)))


class TestSaveLoad():
    @staticmethod
    def compare_models(model_1, model_2):
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if not torch.equal(key_item_1[1], key_item_2[1]):
                if (key_item_1[0] == key_item_2[0]):
                    return False
                else:
                    raise Exception
        return True

    def test_save_model(self, tmpdir, model_and_meta):
        model_name, global_step, saved_model = model_and_meta
        saved_optimizer = torch.optim.Adagrad(saved_model.parameters(), lr=5e-4)
        save_network(saved_model,
                     saved_optimizer,
                     tmpdir,
                     model_name,
                     global_step)
        assert ('%s_%s.pth' % (model_name, global_step) in listdir(tmpdir))

    def test_save_model_gpu(self, tmpdir, model_and_meta):
        model_name, global_step, saved_model = model_and_meta
        saved_model = DummyTorchModule().cuda()
        optimizer = torch.optim.Adagrad(saved_model.parameters(), lr=5e-4)
        save_network(saved_model,
                     optimizer,
                     tmpdir,
                     model_name,
                     global_step)

        assert ('%s_%s.pth' % (model_name, global_step) in listdir(tmpdir))

    def test_save_load(self, tmpdir, model_and_meta):
        # test if load saved model we get same result
        model_name, global_step, saved_model = model_and_meta
        saved_optimizer = torch.optim.Adagrad(saved_model.parameters(), lr=5e-4)

        save_network(saved_model,
                     saved_optimizer,
                     tmpdir,
                     model_name,
                     global_step,
                     use_gpu=False, )
        reload_model = DummyTorchModule()
        reload_optimizer = torch.optim.Adagrad(reload_model.parameters(), lr=5e-4)
        assert (not self.compare_models(saved_model, reload_model))
        reload_model, reload_optimizer = load_network(reload_model,
                                                      reload_optimizer,
                                                      tmpdir,
                                                      model_name,
                                                      global_step,
                                                      use_gpu=False,
                                                      reset_optimizer=False)
        assert (self.compare_models(saved_model, reload_model))

    def test_save_load_gpu(self, tmpdir, model_and_meta):
        # test if load saved model we get same result
        model_name, global_step, saved_model = model_and_meta
        saved_model = saved_model.cuda()
        saved_optimizer = torch.optim.Adagrad(saved_model.parameters(), lr=5e-4)

        reload_model = DummyTorchModule().cuda()
        reload_optimizer = torch.optim.Adagrad(reload_model.parameters(), lr=5e-4)

        save_network(saved_model,
                     saved_optimizer,
                     tmpdir,
                     model_name,
                     global_step,
                     use_gpu=True)

        # it's close to impossible that two initialized parameter matrix is exactly the same
        assert (not self.compare_models(saved_model, reload_model))
        reload_model, reload_optimizer = load_network(reload_model,
                                                      reload_optimizer,
                                                      tmpdir,
                                                      model_name,
                                                      global_step,
                                                      use_gpu=True,
                                                      reset_optimizer=False)
        # test if reload model is the same as test model
        assert (self.compare_models(saved_model, reload_model))

    def test_save_load_last_checkpoint(self, tmpdir):
        model_name = 'Dummy'
        global_step_0 = 0
        saved_model_0 = DummyTorchModule()
        saved_optimizer_0 = torch.optim.Adagrad(saved_model_0.parameters(), lr=5e-4)

        global_step_1 = 1
        saved_model_1 = DummyTorchModule()
        saved_optimizer_1 = torch.optim.Adagrad(saved_model_1.parameters(), lr=5e-4)

        reload_model = DummyTorchModule()
        reload_optimizer = torch.optim.Adagrad(reload_model.parameters(), lr=5e-4)
        save_network(saved_model_0,
                     saved_optimizer_0,
                     tmpdir,
                     model_name,
                     global_step_0,
                     use_gpu=False)

        save_network(saved_model_1,
                     saved_optimizer_1,
                     tmpdir,
                     model_name,
                     global_step_1,
                     use_gpu=False)

        reload_model, reload_optimizer, global_step = load_last_checkpoint(reload_model,
                                                              reload_optimizer,
                                                              tmpdir,
                                                              model_name,
                                                              use_gpu=False)
        assert (global_step == 1)
        assert (not self.compare_models(saved_model_0, reload_model))
        assert (self.compare_models(saved_model_1, reload_model))

    def test_invalid_model_name(self, model_and_meta, tmpdir):
        model_name, global_step, saved_model = model_and_meta
        model_name = 'dummy_model'
        saved_optimizer = torch.optim.Adagrad(saved_model.parameters(), lr=5e-4)

        with pytest.raises(ValueError, match='model name can not contain "." or "_"'):
            save_network(saved_model,
                         saved_optimizer,
                         tmpdir,
                         model_name,
                         global_step=False,
                         use_gpu=True)

        with pytest.raises(ValueError, match='model name can not contain "." or "_"'):
            load_last_checkpoint(saved_model,
                                 saved_optimizer,
                                 tmpdir,
                                 model_name,
                                 use_gpu = False,
                                 reset_optimizer=False)

        with pytest.raises(ValueError, match='model name can not contain "." or "_"'):
            load_network(saved_model,
                         saved_optimizer,
                         tmpdir,
                         model_name,
                         global_step,
                         use_gpu = False,
                         reset_optimizer=False)


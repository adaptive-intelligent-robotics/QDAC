import numpy as np
import torch

from attrdict import AttrDict
from time import time
from baselines.PPGA.utils.utilities import log
from baselines.PPGA.models.vectorized import VectorizedActor
from baselines.PPGA.models.actor_critic import Actor, PGAMEActor

TEST_CFG = AttrDict({
    'normalize_obs': False,
    'normalize_rewards': False,
    'obs_dim': 227,
    'action_shape': np.array(17),
    'env_cfg': {
        'env_name': 'humanoid',
        'seed': 0,
        'num_dims': 2,
        'env_batch_size': None,
    }
})


def test_serialize_deserialize_pgame_actor():
    obs_size, action_shape = 87, (8,)
    agent1 = PGAMEActor(obs_shape=obs_size, action_shape=action_shape)
    agent1_params = agent1.serialize()

    agent2 = PGAMEActor(obs_shape=obs_size, action_shape=action_shape).deserialize(agent1_params)
    agent2_params = agent2.serialize()
    assert np.allclose(agent1_params, agent2_params)


def test_vectorized_policy():
    global TEST_CFG
    obs_dim, action_shape = TEST_CFG.obs_dim, TEST_CFG.action_shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_models = 10
    models = [Actor(obs_dim, action_shape, False, False).to(device) for _ in range(num_models)]
    vec_model = VectorizedActor(models, Actor, obs_dim, action_shape, False, False).to(device)
    obs = torch.randn((num_models, obs_dim)).to(device)

    # test same number of models as number of obs
    res_for_loop = []
    start_for = time()
    for o, model in zip(obs, models):
        out = model(o)
        res_for_loop.append(out)
    res_for_loop = torch.cat(res_for_loop)
    elapsed_for = time() - start_for

    start_vec = time()
    res_vectorized = vec_model(obs).flatten().to(torch.float32)
    elapsed_vec = time() - start_vec

    assert torch.allclose(res_for_loop, res_vectorized, atol=1e-3), "Error! The vectorized policy does not produce the " \
                                                         "same outputs as naive for-loop over all the individual models"

    print(f'For loop over models took {elapsed_for:.2f} seconds. Vectorized inference took {elapsed_vec:.2f} seconds')

    # test multiple obs per model
    num_models = 7
    num_obs = num_models * 3

    models = [Actor(obs_dim, action_shape, False, False).to(device) for _ in range(num_models)]
    vec_model = VectorizedActor(models, Actor, obs_dim, action_shape, False, False).to(device)
    obs = torch.randn((num_obs, obs_dim)).to(device)

    with torch.no_grad():
        res_vectorized = vec_model(obs)
        res_for_loop = []
        obs = obs.reshape(num_models, -1, obs_dim)
        for next_obs, model in zip(obs, models):
            obs1, obs2, obs3 = next_obs[0].reshape(1, -1), next_obs[1].reshape(1, -1), next_obs[2].reshape(1, -1)
            res_for_loop.append(model(obs1))
            res_for_loop.append(model(obs2))
            res_for_loop.append(model(obs3))

    res_for_loop = torch.cat(res_for_loop).flatten()
    res_vectorized = res_vectorized.flatten().to(torch.float32)

    assert torch.allclose(res_for_loop, res_vectorized, atol=1e-1)


# from https://gist.github.com/rohan-varma/a0a75e9a0fbe9ccc7420b04bff4a7212
def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        log.info(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
            model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            log.info(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            log.info(f"Tensor mismatch: {v_1} vs {v_2}")
            return False
    return True


def all_params_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def test_vectorized_to_list():
    '''Make sure the models_list() function returns the list of models to the exact
    same state they were passed in'''
    obs_shape, action_shape = TEST_CFG.obs_dim, TEST_CFG.action_shape
    models = [Actor(obs_shape, action_shape, False, False) for _ in range(10)]
    vec_model = VectorizedActor(models, Actor, obs_shape=obs_shape, action_shape=action_shape)
    models_returned = vec_model.vec_to_models()

    for m_old, m_new in zip(models, models_returned):
        m_old = m_old.cpu()
        m_new = m_new.cpu()
        old_statedict, new_statedict = m_old.state_dict(), m_new.state_dict()
        assert validate_state_dicts(old_statedict, new_statedict), "Error: State dicts for original model and model" \
                                                                   " returned by the vectorized model are not the same"

        # double check all parameters are the same
        assert all_params_equal(m_old, m_new), "Error: not all parameters are the same for the original and returned " \
                                               "model"


if __name__ == '__main__':
    pass

# test_ntm_with_modern_training_runs.py
import unittest
import unittest.mock as mock
import sys
import os

import torch
import numpy as np

import ntm_with_modern_training_runs as mod


class TestPickGpuWithMostFreeMem(unittest.TestCase):
    def test_pick_gpu_with_most_free_mem_runs(self):
        try:
            idx = mod.pick_gpu_with_most_free_mem()
            self.assertIsInstance(idx, int)
            self.assertGreaterEqual(idx, 0)
        except Exception as e:
            self.fail(f"pick_gpu_with_most_free_mem() raised an exception: {e}")


class TestVocab(unittest.TestCase):
    def test_get_char_vocab(self):
        vocab_list, char_to_id, id_to_char = mod.get_char_vocab()
        self.assertIn('<PAD>', vocab_list)
        self.assertIn('<bos>', vocab_list)
        self.assertIn('<eos>', vocab_list)
        self.assertGreater(len(vocab_list), 10)  # digits + letters + special
        # Check mapping consistency
        self.assertEqual(id_to_char[char_to_id['<PAD>']], '<PAD>')


class TestStrToTensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.char_to_id, _ = mod.get_char_vocab()

    def test_empty_strings(self):
        out = mod.str_to_tensor([], self.char_to_id, 10)
        self.assertEqual(out.shape, (0, 10))

    def test_some_strings(self):
        batch_strs = ["ABC", "HELLO"]
        out = mod.str_to_tensor(batch_strs, self.char_to_id, max_seq_len=5)
        self.assertEqual(out.shape, (2, 5))
        self.assertEqual(out.dtype, torch.long)

    def test_truncation(self):
        batch_strs = ["ABCDE"]
        out = mod.str_to_tensor(batch_strs, self.char_to_id, 3)
        self.assertEqual(out.shape, (1,3))

    def test_zero_max_seq_len(self):
        # if max_seq_len=0, we expect shape (N, 0)
        batch_strs = ["ABC", "DEF"]
        out = mod.str_to_tensor(batch_strs, self.char_to_id, 0)
        self.assertEqual(out.shape, (2, 0))


class TestShiftByOnePairs(unittest.TestCase):
    def test_simple_case(self):
        x_str, y_str = mod.shift_by_one_pairs("ABC", "XYZ")
        self.assertEqual(x_str, "<bos>ABC")
        self.assertEqual(y_str, "XYZ<eos>")


class TestGenerators(unittest.TestCase):
    def test_generate_copy_task_str(self):
        in_list, out_list = mod.generate_copy_task_str(5, 3, train=True)
        self.assertEqual(len(in_list), 5)
        self.assertEqual(len(out_list), 5)

    def test_generate_copy_task_str_zero_context(self):
        in_list, out_list = mod.generate_copy_task_str(2, 0, train=True)
        # They should still produce strings with just <bos> and <eos>
        for i, o in zip(in_list, out_list):
            self.assertTrue(i.startswith("<bos>"))
            self.assertTrue(o.endswith("<eos>"))

    def test_generate_repeat_copy_task_str(self):
        in_list, out_list = mod.generate_repeat_copy_task_str(5, 3, repeat_min=1, repeat_max=2)
        self.assertEqual(len(in_list), 5)
        self.assertEqual(len(out_list), 5)

    def test_generate_associative_recall_task_str(self):
        in_list, out_list = mod.generate_associative_recall_task_str(4, item_len=2, num_items=2)
        self.assertEqual(len(in_list), 4)
        self.assertEqual(len(out_list), 4)

    def test_generate_arithmetic_task_str(self):
        in_list, out_list = mod.generate_arithmetic_task_str(5, context_length=2, task_type="add", max_num=3)
        self.assertEqual(len(in_list), 5)
        self.assertEqual(len(out_list), 5)

    def test_generate_fibonacci_task_str(self):
        in_list, out_list = mod.generate_fibonacci_task_str(3, context_length=5, max_n=8, train=True)
        self.assertEqual(len(in_list), 3)
        self.assertEqual(len(out_list), 3)

    def test_generate_factorial_task_str(self):
        in_list, out_list = mod.generate_factorial_task_str(3, context_length=5, max_n=4, train=True)
        self.assertEqual(len(in_list), 3)
        self.assertEqual(len(out_list), 3)


class TestModelsNTM(unittest.TestCase):
    def setUp(self):
        self.model = mod.NTM(input_size=8, output_size=10, hidden_size=16, memory_size=8, head_size=4, num_heads=1)
        self.model.eval()

    def test_forward(self):
        x_emb = torch.randn(2, 5, 8)
        out, mem, hid = self.model(x_emb)
        self.assertEqual(out.shape, (2, 5, 10))
        self.assertEqual(mem.shape, (2, 8, 4))
        self.assertIsInstance(hid, tuple)
        self.assertEqual(len(hid), 2)


class TestModelsDNC(unittest.TestCase):
    def setUp(self):
        self.model = mod.DNC(input_size=8, output_size=10, hidden_size=16,
                             memory_size=8, head_size=4, num_heads=1)
        self.model.eval()

    def test_forward(self):
        x_emb = torch.randn(2, 5, 8)
        out, mem_state, hid = self.model(x_emb)
        self.assertEqual(out.shape, (2, 5, 10))
        self.assertIsInstance(mem_state, tuple)
        self.assertEqual(len(mem_state), 4)
        self.assertIsInstance(hid, tuple)
        self.assertEqual(len(hid), 2)


class TestModelsTransformerNTM(unittest.TestCase):
    def setUp(self):
        self.model = mod.TransformerNTM(input_size=8, output_size=10, hidden_size=16)
        self.model.eval()

    def test_forward(self):
        x_emb = torch.randn(2, 5, 8)
        out, mem, hid = self.model(x_emb)
        self.assertEqual(out.shape, (2, 5, 10))
        self.assertIsNone(mem)
        self.assertIsNone(hid)


class TestGroupParamsByLayer(unittest.TestCase):
    def test_grouping(self):
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
                self.layer1_extra = nn.Linear(10, 10)
                self.other = nn.Linear(10, 5)

        dmy = Dummy()
        named_params = list(dmy.named_parameters())
        grouped = mod.group_params_by_layer(named_params)
        self.assertIn('layer1', grouped)
        self.assertIn('layer1_extra', grouped)
        self.assertIn('other', grouped)


class TestMezoSingle(unittest.TestCase):
    def test_mezo_char_single(self):
        model = nn.Sequential(nn.Linear(4,4))
        x = torch.randn(2,3,4)
        y = torch.randint(0,4,(2,3))
        criterion = nn.CrossEntropyLoss()
        loss_val = mod.mezo_char_single(model, x, y, criterion, epsilon=1e-3)
        self.assertIsInstance(loss_val, float)


class TestMezoLayerwise(unittest.TestCase):
    def test_mezo_char_layerwise(self):
        model = nn.Sequential(
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4,2)
        )
        x = torch.randn(2,3,4)
        y = torch.randint(0,2,(2,3))
        criterion = nn.CrossEntropyLoss()
        loss_val = mod.mezo_char_layerwise(model, x, y, criterion, epsilon=1e-3)
        self.assertIsInstance(loss_val, float)


class TestMaybeUpdateCurriculum(unittest.TestCase):
    def test_update_curriculum_copy(self):
        # test the function maybe_update_curriculum directly
        consecutive_succ = mod.main.__globals__['consecutive_succ']
        # but we can't set it directly if it's declared local
        # Instead, we replicate the logic or do a partial patch.

        # We'll define a local version that calls the actual function
        local_func = mod.main.__globals__['maybe_update_curriculum']

        # For copy
        # start with context=2, success>0.95 => +1
        cur_ct, cur_mn = 2, 10
        # pass threshold 5 times
        for _ in range(5):
            new_ct, new_mn = local_func(0.96, cur_ct, cur_mn, 'copy')
            cur_ct, cur_mn = new_ct, new_mn
        self.assertGreater(cur_ct, 2)

    def test_update_curriculum_add(self):
        local_func = mod.main.__globals__['maybe_update_curriculum']
        cur_ct, cur_mn = 10, 5
        # pass threshold 5 times
        for _ in range(5):
            new_ct, new_mn = local_func(0.99, cur_ct, cur_mn, 'add')
            cur_ct, cur_mn = new_ct, new_mn
        self.assertGreater(cur_mn, 5)

    def test_no_update_curriculum(self):
        # if train_acc <0.95 => no update
        local_func = mod.main.__globals__['maybe_update_curriculum']
        cur_ct, cur_mn = 10, 5
        new_ct, new_mn = local_func(0.50, cur_ct, cur_mn, 'copy')
        self.assertEqual(new_ct, 10)
        self.assertEqual(new_mn, 5)


class TestGenerateTaskData(unittest.TestCase):
    def test_generate_task_data_copy(self):
        local_func = mod.main.__globals__['generate_task_data']
        in_list, out_list = local_func(5, 'copy', context_len=2, maxn=10, train=True)
        self.assertEqual(len(in_list), 5)
        self.assertEqual(len(out_list), 5)

    def test_generate_task_data_add(self):
        local_func = mod.main.__globals__['generate_task_data']
        in_list, out_list = local_func(5, 'add', context_len=2, maxn=5, train=True)
        self.assertEqual(len(in_list), 5)
        self.assertEqual(len(out_list), 5)

    def test_unknown_task(self):
        local_func = mod.main.__globals__['generate_task_data']
        with self.assertRaises(ValueError):
            local_func(5, 'XYZ', 2, 5, True)


class TestTrainMicroBatch(unittest.TestCase):
    def test_train_micro_batch_mezo_single(self):
        # We'll pass a minimal model & a mock arg
        parser = mod.argparse.ArgumentParser()
        parser.add_argument("--mezo", action="store_true")
        parser.add_argument("--mezo_layerwise", action="store_true")
        # we won't parse full, just do a partial
        class Args: pass
        args = Args()
        args.mezo = True
        args.mezo_layerwise = False

        # We'll call train_micro_batch via a local reference
        local_func = mod.main.__globals__['train_micro_batch']

        # minimal model
        model = nn.Sequential(nn.Linear(4,4))
        # set up global "model" in main's scope if needed
        mod.main.__globals__['model'] = model
        # likewise for criterion, ...
        crit = nn.CrossEntropyLoss(ignore_index=0)
        mod.main.__globals__['criterion'] = crit

        # we also need embed in global
        embed = nn.Embedding(10,4)
        mod.main.__globals__['embed'] = embed

        # we also need "optimizer" in global
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        mod.main.__globals__['optimizer'] = optimizer

        x_emb = torch.randn(2,3,4)
        y_ids = torch.randint(0,4,(2,3))

        # Actually call it
        loss_val = local_func(x_emb, y_ids)
        self.assertIsInstance(loss_val, float)


class TestMainFunction(unittest.TestCase):
    @mock.patch.object(sys, 'argv', ["prog",
                                     "--task=copy",
                                     "--arch=ntm",
                                     "--context_length=2",
                                     "--max_seq_len=5",
                                     "--micro_batch_size=1",
                                     "--macro_batch_size=1",
                                     "--max_iters=1",
                                     "--mezo",
                                     "--epsilon=1e-3",
                                     "--wandb_proj="])
    def test_main_minimal(self):
        try:
            mod.main()
        except SystemExit:
            pass
        except Exception as e:
            self.fail(f"main() raised an exception unexpectedly: {e}")


if __name__ == '__main__':
    unittest.main()

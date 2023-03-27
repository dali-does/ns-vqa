import unittest
import json

from executors import ClevrExecutor
import utils.utils as utils
import utils.preprocess as preprocess
from tools.preprocess_questions import program_to_str

class TestOriginalClevr(unittest.TestCase):
    data_path = 'data'

    def setUp(self):
        scene_json = '{}/val-scenes.json'.format(self.data_path)
        vocab_json = '{}/vocab-orig.json'.format(self.data_path)
        self.executor = ClevrExecutor(scene_json, scene_json, vocab_json)

    def test_same_shape(self):
        # exist, same_shape, unique, filter_material[metal],
        # filter_size[large], scene
        x = [6, 24, 28, 11, 16, 26, 2]
        index = 0
        split = 'val'
        ans = self.executor.run(x,index,split)
        self.assertEquals(ans, 'no')

class TestMultihop(unittest.TestCase):
    data_path = 'data'

    def setUp(self):
        scene_json = '{}/val-scenes.json'.format(self.data_path)
        vocab_json = '{}/vocab.json'.format(self.data_path)
        self.executor = ClevrExecutor(scene_json, scene_json, vocab_json)

    def test_execute_onehop(self):
        # count, subtraction_set, filter_shape[cube],
        # filter_color[gray], scene, end
        x = [4, 17, 13, 8, 16, 2]
        index = 0
        split = 'val'
        ans = self.executor.run(x,index,split)
        self.assertEquals(ans, '3')

    def test_execute_subtract_multihop(self):
        # count, subtraction_set, filter_shape[cylinder], filter_color[brown],
        # subtraction_set, filter_shape[cube], filter_color[gray], scene, end
        x = [4, 17, 14, 6, 17, 13, 8, 16, 2]
        index = 0
        split = 'val'
        ans = self.executor.run(x,index,split)
        self.assertEquals(ans, '2')

    def test_subtract_empty_set(self):
        # count, subtraction_set, filter_shape[cube], filter_color[cyan],
        # subtraction_set, filter_shape[sphere], filter_color[gray], scene, end
        x = [4, 17, 13, 7, 17, 15, 8, 16, 2]
        index = 0
        split = 'val'
        ans = self.executor.run(x,index,split)
        self.assertEquals(ans, '5')

class TestAttributes(unittest.TestCase):
    data_path = 'data'

    def setUp(self):
        scene_json = '{}/val-scenes.json'.format(self.data_path)
        vocab_json = '{}/vocab-remove.json'.format(self.data_path)
        self.executor = ClevrExecutor(scene_json, scene_json, vocab_json)

    def test_encode_remove(self):
        x = [6, 14, 13, 7, 14, 12, 16, 15, 2]
        program = [ 'count','remove', 'red', 'cube', 'remove', 
                'purple', 'sphere', 'scene', '<END>']
        vocab_json = '{}/vocab-remove.json'.format(self.data_path)
        vocab = utils.load_vocab(vocab_json)
        tokens = preprocess.encode(program, vocab['program_token_to_idx'])

        self.assertEquals(tokens, x)

    def test_encode_original(self):
        program = [ 'exist', 'same_shape', 'unique', 'filter_material', 
                'metal', 'filter_size', 'large', 'scene', '<END>']
        x = [18, 19, 20, 21, 23, 22, 24, 15, 2]
        vocab_json = '{}/vocab-remove.json'.format(self.data_path)
        vocab = utils.load_vocab(vocab_json)
        tokens = preprocess.encode(program, vocab['program_token_to_idx'])

        self.assertEquals(tokens, x)

    def test_remove_two(self):
        # count, remove, red, cube, remove, purple, sphere, scene, end
        x = [6, 14, 13, 7, 14, 12, 16, 15, 2]
        index = 0
        split = 'val'
        ans = self.executor.run(x,index,split, use_attributes=True)
        self.assertEquals(ans, '4')

    def test_remove_many(self):
        # count, remove, cylinder, remove, purple, sphere, scene, end
        x = [6, 14, 9, 14, 12, 16, 15, 2]
        index = 3
        split = 'val'
        ans = self.executor.run(x,index,split, use_attributes=True)
        self.assertEquals(ans, '2')

    def test_attribute_with_exist_sameshape_unique_filter(self):
        # exist, same_shape, unique, filter_material, metal,
        # filter_size, large, scene
        x = [18, 19, 20, 21, 23, 22, 24, 15, 2]
        index = 0
        split = 'val'
        ans = self.executor.run(x,index,split, use_attributes=True)
        self.assertEquals(ans, 'no')

    def test_whole_chain_with_attributes_original(self):
        vocab_json = '{}/vocab-remove.json'.format(self.data_path)
        vocab = utils.load_vocab(vocab_json)

        f = open('{}/orig-val.json'.format(self.data_path))
        question = json.load(f)['questions'][0]
        mode = vocab['mode']
        program = program_to_str(question['program'], mode)
        program = preprocess.tokenize(program)
        tokens = preprocess.encode(program, vocab['program_token_to_idx'])
        x = [1, 18, 19, 20, 21, 23, 22, 24, 15, 2]
        f.close()

        self.assertEquals(tokens, x)

if __name__ == '__main__':
    unittest.main()

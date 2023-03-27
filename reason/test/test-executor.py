import unittest
from executors import ClevrExecutor

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


if __name__ == '__main__':
    unittest.main()

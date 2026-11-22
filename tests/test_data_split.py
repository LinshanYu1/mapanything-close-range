import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from close_range.data_split import make_split, load_split


class SplitTests(unittest.TestCase):
    def test_mixed_sources_grouping_repeatability_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source in ['phone', 'graspnet']:
                (root / source).mkdir()
            for source, names in [('phone', [f'scene-{i:03}' for i in range(1, 21)]),
                                  ('graspnet', [f'scene_{i:04}' for i in range(190)])]:
                for name in names:
                    data = dict(source=source, scene_id=name, original_scene=name,
                                views=[dict(sha256=source+name)])
                    (root/source/f'{name}.json').write_text(json.dumps(data))
            twin = dict(source='phone', scene_id='different_camera', original_scene='scene-002',
                        views=[dict(sha256='extra')])
            (root/'phone/extra.json').write_text(json.dumps(twin))
            with patch('close_range.training.dataset.SceneDistanceDataset'):
                first = make_split([root/'graspnet',root/'phone'],root/'split.json')
                second = make_split([root/'phone',root/'graspnet'],root/'split2.json')
                for k in ['train','val','test']:
                    self.assertEqual(first[k],second[k])
                self.assertEqual(first['metadata']['scene_counts']['graspnet'],dict(train=80,val=20,test=90))
                self.assertEqual(first['metadata']['scene_counts']['phone'],dict(train=14,val=3,test=3))
                with self.assertRaises(FileExistsError):
                    make_split([root/'phone'],root/'split.json')
                with self.assertRaises(FileNotFoundError):
                    make_split([root/'missing'],root/'bad.json')
            load_split(root/'split.json')
            for paths in first.values():
                if isinstance(paths,list) and any(p.endswith('/extra.json') for p in paths):
                    self.assertTrue(any(p.endswith('/scene-002.json') for p in paths))

    def test_validation_failure_does_not_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'phone').mkdir()
            (root/'phone/a.json').write_text(json.dumps(dict(source='phone',scene_id='scene-001')))
            with patch('close_range.training.dataset.SceneDistanceDataset',side_effect=ValueError('bad labels')):
                with self.assertRaisesRegex(ValueError,'bad labels'):
                    make_split([root/'phone'],root/'split.json')
            self.assertFalse((root/'split.json').exists())

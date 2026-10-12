"""Replication scheduling/provenance/resume tests; synthetic data only."""
import contextlib
import io
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from . import run_paired_replication as runner
from . import run_paired_window as original
from .test_paired_window import ToyModel, ToyTrain, scratch


class ReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_training_body_unchanged_except_evaluation_control_and_log_prefix(self):
        source=inspect.getsource(original.train_group)
        expected=(source.replace("if method!='B0':","if method!='PAIR':")
                  .replace("f'nyuv2/B0_seed{seed}/curves/{condition}.json'",
                           "f'nyuv2/PAIR_seed{seed}/curves/{condition}.json'")
                  .replace('PAIRED_','REPLICATION_'))
        self.assertEqual(expected,inspect.getsource(runner.train_group))
        self.assertIs(original.backward_objective,runner.backward_objective)
        self.assertIs(original.PairedCMX,runner.PairedCMX)

    def test_complete_matrix(self):
        root=scratch()
        args=SimpleNamespace(output_dir=root,budget_hours=6,reference_results=[
            dict(dataset=d,method=m,seed=42,elapsed_minutes=45)
            for m in runner.METHODS for d in ('nyuv2','private')])
        calls=[]
        def train(a,d,m,s,p):
            calls.append((d,m,s));return dict(dataset=d,method=m,seed=s)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner,'preflight',lambda a:{}))
            stack.enter_context(patch.object(runner,'train_group',train))
            stack.enter_context(patch.object(runner,'summarize',lambda *a:None))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            runner.execute(args)
        self.assertEqual(calls,[(d,m,s) for s in (777,2025) for m in ('PAIR','CONS') for d in ('nyuv2','private')])
        result=runner.read(root/'complete.json')
        self.assertEqual(result['training_groups'],8)
        self.assertEqual(result['epochs_per_group'],20)
        self.assertEqual(result['combined_seeds'],[42,777,2025])
        self.assertEqual(result['public_condition_evaluations'],88)
        self.assertFalse(result['final_test_access'])

    def test_matched_summary_excludes_incomplete_seed_and_uses_sample_std(self):
        root=scratch();args=SimpleNamespace(output_dir=root,reference_results=[])
        def row(s,m,v):
            return dict(dataset='nyuv2',method=m,seed=s,clean_miou=v,corruption_mean_miou=v,
                        photon_proxy_heavy_miou=v,joint_heavy_miou=v)
        args.reference_results=[row(42,'PAIR',.5),row(42,'CONS',.51)]
        rows=[row(777,'PAIR',.6),row(777,'CONS',.63),row(2025,'PAIR',.8)]
        with contextlib.redirect_stdout(io.StringIO()):
            runner.summarize(args,rows)
        result=runner.read(root/'comparison.json')
        selected=next(r for r in result['summaries'] if r['dataset']=='nyuv2' and r['method']=='CONS')
        self.assertEqual(selected['seeds'],[42,777])
        self.assertAlmostEqual(selected['clean_miou']['mean'],.57)
        self.assertAlmostEqual(selected['clean_miou']['paired_delta_pp'],2)
        self.assertAlmostEqual(selected['clean_miou']['paired_delta_std_pp'],2**.5)
        self.assertEqual(len(result['paired_comparisons']),2)

    def test_empty_tiny_metric_preserved(self):
        args=SimpleNamespace(output_dir=scratch(),reference_results=[])
        rows=[dict(dataset='private',method=m,seed=777,metrics=dict(
            foreground_iou=.7,boundary_f1_1px=.8,background_fp_frame_rate=.1,
            tiny_component_recall_iou50_area80=None)) for m in runner.METHODS]
        with contextlib.redirect_stdout(io.StringIO()):
            runner.summarize(args,rows)
        data=runner.read(args.output_dir/'comparison.json')
        self.assertIsNone(data['paired_comparisons'][0]['delta_pp']['tiny_component_recall_iou50_area80'])

    def reference_fixture(self):
        root=scratch();args=SimpleNamespace(reference_dir=root)
        runner.save(root/'protocol.json',dict(version='paired_v1',epochs=20,fingerprints={'source':'same'},code={}))
        runner.save(root/'complete.json',dict(epochs_per_group=20,final_test_access=False))
        for d in ('private','nyuv2'):
            for m in runner.METHODS:
                folder=root/d/f'{m}_seed42'
                result=dict(dataset=d,method=m,seed=42,epochs=20,checkpoint_sha256='checkpoint')
                if d=='nyuv2':
                    result['scores']=[dict(condition=f'{f}_{v}',miou=.4) for f,v in runner.CONDITIONS]
                    for score in result['scores']:
                        runner.save(folder/'curves'/f'{score["condition"]}.json',
                                    dict(checkpoint_sha256='checkpoint',regions={'all':{'miou':.4}}))
                runner.save(folder/'result.json',result)
                runner.save(folder/'history.json',dict(epochs=[dict(epoch=e) for e in range(1,21)]))
        return args

    def test_reference_validation(self):
        args=self.reference_fixture()
        files,rows=runner.verify_reference(args,{'source':'same'})
        self.assertEqual(len(rows),4);self.assertEqual(len(files),54)
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            runner.verify_reference(args,{'source':'changed'})
        path=args.reference_dir/'nyuv2/PAIR_seed42/curves/clean_medium.json'
        runner.save(path,dict(checkpoint_sha256='wrong',regions={'all':{'miou':.4}}))
        with self.assertRaisesRegex(ValueError,'disagreement'):
            runner.verify_reference(args,{'source':'same'})

    def test_reference_wrong_epoch_and_code_rejected(self):
        args=self.reference_fixture()
        path=args.reference_dir/'protocol.json';data=runner.read(path)
        data['epochs']=16;runner.save(path,data)
        with self.assertRaisesRegex(ValueError,'at20epochs'):
            runner.verify_reference(args,{'source':'same'})
        data['epochs']=20;data['code']={'paper_benchmark_v1/paired_model.py':'changed'}
        runner.save(path,data)
        with self.assertRaisesRegex(ValueError,'dependency changed'):
            runner.verify_reference(args,{'source':'same'})

    def test_resume_and_pair_evaluation_without_B0(self):
        root=scratch();args=SimpleNamespace(output_dir=root,private_dir=root,nyuv2_dir=root,device='cpu')
        source=dict(model=ToyModel('PAIR').state_dict(),args={'model':'CMX_B2','decoder_dim':32})
        metric=dict(regions={'all':{'miou':.4}},degradation_audit=[])
        with contextlib.ExitStack() as stack:
            for key,value in [('load_source',lambda *a:source),('PairedCMX',ToyModel),('TrainingData',ToyTrain),
                ('PrivateTrain',ToyTrain),('PrivatePaperDataset',lambda *a:[]),
                ('evaluate',lambda *a:{'foreground_iou':.4}),('private_details',lambda *a:{'rows':[]}),
                ('DiagnosticData',lambda *a:None),('regional',lambda *a:metric),('EPOCHS',2)]:
                stack.enter_context(patch.object(runner,key,value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            checkpoint=runner.atomic_checkpoint
            def stop(path,state):
                checkpoint(path,state);raise RuntimeError('synthetic interruption')
            with patch.object(runner,'atomic_checkpoint',stop):
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                    runner.train_group(args,'nyuv2','PAIR',777,{'test':1})
            with self.assertRaisesRegex(ValueError,'protocol/group'):
                runner.train_group(args,'nyuv2','PAIR',777,{'test':2})
            for seed in runner.SEEDS:
                for method in runner.METHODS:
                    for dataset in ('nyuv2','private'):
                        result=runner.train_group(args,dataset,method,seed,{'test':1})
                        self.assertEqual(result,runner.train_group(args,dataset,method,seed,{'test':1}))


if __name__=='__main__':
    unittest.main()

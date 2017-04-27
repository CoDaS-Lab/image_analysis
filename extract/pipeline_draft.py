import numpy as np
from feature import Feature

class Pipeline:
    def __init__(self, ops=None, data=None, seq=None, save_all=None,
                 models=None):
        self.data = data
        self.models = models      # Should be a list of models.
        self.save_all = save_all  # True saves all, False saves none, and
                                  # None uses defaults set by user.

        self.batch_ops = None
        self.frame_ops = None
        self.seq_ops = None
        self.set_op_lists(ops, seq)

        self.empty_frame = {}           # Defined by set method below.
        self.output = []          # Output data structure of the pipeline.


    def set_ops(self, ops, seq):
        batch_ops = []
        frame_ops = []
        if ops is not None:
            for op in ops:
                assert isinstance(op, Feature)
                if op.batch_op:
                    batch_ops.append(op)
                elif op.frame_op:
                    frame_ops.append(op)
                else:
                    raise ValueError('One and only one of either batch op or '
                                      + 'or frame op is allowed to be True!')
        elif ops is None or ops == []:
            batch_ops = []
            frame_ops = []

        if seq is not None:
            for op in seq:
                assert isinstance(op, Feature)
        elif seq is None or seq == []:
            seq = []
        
        self.batch_ops = batch_ops
        self.frame_ops = frame_ops
        self.seq_ops = seq


    def reset_ops(self, ops=None, seq=None):
        self.set_ops(ops, seq)

    
    def set_batch_ops(self, batch_ops=None):
        if batch_ops is None:
            batch_ops = []

        for op in batch_ops:
            assert isinstance(op, Feature)
            assert op.batch_op

        self.batch_ops = batch_ops


    def set_frame_ops(self, frame_ops=None):
        if frame_ops is None:
            frame_ops = []
        
        for op in frame_ops:
            assert isinstance(op, Feature)
            assert op.frame_op

        self.frame_ops = frame_ops


    def set_seq(self, seq=[]):
        for op in seq_ops:
            assert isinstance(Feature)
        # Need to handle case of someone sending a batch into frame op & vise versa
        self.seq_ops = seq_ops


    def set_empty_frame(self, batch_ops, frame_ops, seq_ops):
        assert isinstance(batch_ops, list)
        assert isinstance(frame_ops, list)
        assert isinstance(seq_ops, list)

        frame = {
            'input': None,
            'meta_data': None,
            'batch_features': {},
            'frame_features': {},
            'seq': None,
            'seq_features': {},
            'seq_output': None}

        for op in batch_features:
            frame['batch_features'].update(op.key: None)
        for op in frame_ops:
            frame['frame_features'].update(op.key: None)
        for op in seq_ops:
            frame['seq_features'].update(op.key: None)

        self.empty_frame = frame


    def extract(self, keep_input_data=True):
        if self.batch_ops == self.frame_ops == self.seq_ops == []
            raise ValueError('No features were specified for extraction.')
        elif self.seq_ops == []:
            self.extract_nonseq(self.data, self.batch_ops, self.frame_ops)
        elif self.seq_ops != []:
            self.extract_seq(self.data, self.seq_ops)
        else:
            raise ValueError('Someone broke the Pipeline class code.')

        if keep_input_data is False:
            self.data = []

    def extract_nonseq(self, data, batch_ops, frame_ops, seq_ops):
        self.set_frame(batch_ops, frame_ops, seq_ops):
        n_frame = 0
        n_batch = 0

        for batch in self.data:
            batch_dict = {}
            for op in batch_ops:
                batch_dict.update({op.key_name: op.extract(batch)})
            
            for frame in batch:
                frame_dict = self.empty_frame
                for op in frame_ops:
                    frame_dict['frame_features'][op.key_name] = op.extract(frame)
                    frame_dict['batch_features'].update(batch_dict)
                frame_dict['meta_data'].update({'frame_number': n_frame,
                                           'batch_number': n_batch})
                self.output.append(frame_dict)
                n_frame += 1
            n_batch += 1


    def extract_sequence(self, seq_ops):
        for op in seq_ops:
            if op.save:
            # Still need to implement something that accoutns for save vars.


    def as_ndarray(self, frame_key=None, batch_key=None, seq_key=None):
        keys = [frame_key] + [batch_key] + [seq_key]
        
        key_count = 0
        for x in keys:
            if x is not None:
                key_count += 1
        if key_count != 1:
            raise ValueError('One and only one of the three keys may be set.')
        
        elif keys[0] is not None:
            asdf
        elif keys[1] is not None:

        
        data = []
        for frame_dict in self.output:
            data.append(self.frame_dict[key])
        
        return np.array(data)


    def predict(self, model=''):
        pass

    def train(self, model=''):
        pass

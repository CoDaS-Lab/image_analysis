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

        self.frame = {}           # Defined by set method below.
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

    
    def set_batch_ops(self, batch_ops=[]):
        for op in batch_ops:
            assert isinstance(op, Feature)
            assert op.batch_op

        self.batch_ops = batch_ops


    def set_frame_ops(self, frame_ops=[]):
        for op in frame_ops:
            assert isinstance(op, Feature)
            assert op.frame_op

        self.frame_ops = frame_ops


    def set_seq(self, seq=[]):
        for op in seq_ops:
            assert isinstance(Feature)
        # Need to handle case of someone sending a batch into frame op & vise versa
        self.seq_ops = seq_ops


    def set_frame(self, batch_ops, frame_ops, seq_ops):
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

        self.frame = frame


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
        frame = self.set_frame(self, batch_ops, frame_ops, seq_ops):
        n_frame = 0
        n_batch = 0

        for batch in self.data:


    def extract_sequence(self, seq_ops):
        for op in seq_ops:
            if op.save:
            # Still need to implement something that accoutns for save vars.


    def as_ndarray(self, key='output'):
        data = []
        for frame_dict in self.output
            data.append(self.output[key])
        
        return np.array(data)


    def predict(self, model=''):
        pass

    def train(self, model=''):
        pass


    def gen_batch_features(batch_list, batch_op_list):
        """
        DESCRIPTION:
            return list of dictionaries, with each dictionary
            representing a batch and containing features
            extracted according to batch_op_list
    
        PARAMS:
            batch_list: list of batches (ndarrays)
            batch_op_list: list of batch_op objects that extract features
                        from batches
    
        RETURN:
            batch_dictionaries: list of dictionaries, each representing a batch
        """
        batch_dictionaries = []
        for batch in batch_list:
            batch_ds = {
                'input': {
                    'batch': batch
                },
                'metadata': {
    
                }
            }
            batch_dictionaries.append(batch_ds)
            for op in batch_op_list:
                batch_ds['input'].update({op.key_name: op.extract(batch)})
        return batch_dictionaries
    
    
    def gen_frame_features(batch_list, frame_op_list):
        """
        DESCRIPTION:
            return list of dictionaries, with each dictionary
            representing a frame and containing features extracted
            according to frame_op_list
    
        PARAMS:
            batch_list: list of batches (numpy arrays)
            batch_op_list: list of batch_op objects that extract features from
                        batches
    
        RETURN:
            frame_dictionaries: list of dictionaries, each representing a frame
        """
        frame_dictionaries = []
        for batch in batch_list:
            for frame in batch:
                frame_ds = {
                    'input': {
                        'frame': frame
    
                    },
                    'metadata': {
                    }
                }
    
                frame_dictionaries.append(frame_ds)
                for op in frame_op_list:
                    frame_ds['input'].update({op.key_name: op.extract(frame)})
        return frame_dictionaries
    
    
    def batch_to_frame_dictionaries(batch_dictionaries):
        """
        DESCRIPTION:
            take in list of batch dictionaries and output list of
            frame dictionaries
    
        PARAMS:
            batch_dictionary: list of dictionaries representing batches (numpy
                            arrays) of frames; each dictionary should have same
                            exact keys, and each batch should be an ndarry
    
        RETURN:
            frame_dictionary: list of dictionaries representing frames
        """
        frame_dictionaries = []
        batch_index = 0
        for batch_ds in batch_dictionaries:
            for frame in batch_ds['input']['batch']:
                frame_ds = {
                    'input': {
                        'frame': frame
    
                    },
                    'metadata': {
                        'batch_index': batch_index
                    }
                }
    
                frame_dictionaries.append(frame_ds)
                for batch_op in batch_ds['input']:
                    if batch_op != 'batch':
                        frame_ds['input'][batch_op] = batch_ds['input'][batch_op]
            batch_index += 1
    
        return frame_dictionaries
    
    
    def extract_features(batch_ops, frame_ops):
        """
        DESCRIPTION:
    
        PARAMS:
    
        RETURN:
            frame_dictioanries: list of dictionaries, each representing a frame
        """
        self.output = []
        count = 0
        for batch in batch_list:
            batch_dictionary = gen_batch_features([batch], batch_ops)
            frame_dictionaries += (batch_to_frame_dictionaries(batch_dictionary))
    
            if len(frame_ops) != 0:
                for frame in batch:
                    frame_dictionary = gen_frame_features([[frame]], frame_ops)
    
                    # add frame features
                    for feat, val in frame_dictionary[0]['input'].items():
                        frame_dictionaries[count]['input'][feat] = val
                    count += 1
        return frame_dictionaries

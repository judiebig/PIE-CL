import numpy as np


class Data():
    def __init__(self, data, method, shuffle=False, max_len=19):
        self.inputs = np.asarray(data[0])
        self.max_len = max_len
        self.targets = np.asarray(data[1])
        self.method = method
        self.shuffle = shuffle
        self.length = len(self.inputs)

    def data_masks(self, data, pad):
        slice_len = [len(session) for session in data]
        max_len = max(slice_len)
        slice = [session + pad*(max_len-len(session)) for session in data]
        masks = [[1]*len(session) + [0]*(max_len-len(session)) for session in data]
        return slice, masks

    def data_masks_max_clip(self, data, pad):
        # use max_len
        slice_len = [min(len(session), self.max_len) for session in data]
        max_len = max(slice_len)
        slice = np.zeros((len(data), max_len)).astype('int64')
        for idx, s in enumerate(data):
            slice[idx, :slice_len[idx]] = s[-slice_len[idx]:]
        masks = [[1]*s_len + [0]*(max_len-s_len) for s_len in slice_len]
        return slice, masks

    def data_masks_max_clip_pos(self, data, pad):
        # use max_len
        slice_len = [min(len(session), self.max_len) for session in data]
        max_len = max(slice_len)
        slice = np.zeros((len(data), max_len)).astype('int64')
        pos_mask = np.zeros((len(data), max_len)).astype('int64')
        for idx, s in enumerate(data):
            slice[idx, :slice_len[idx]] = s[-slice_len[idx]:]
            pos_mask[idx, :slice_len[idx]] = range(slice_len[idx],0,-1)
        masks = [[1]*s_len + [0]*(max_len-s_len) for s_len in slice_len]
        return slice, masks, pos_mask

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            # self.masks = self.masks[shuffled_arg]  # same with LGSR
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        if self.method == 'normal':
            inputs, masks = self.data_masks(self.inputs[index], [0])
            return inputs, masks, self.targets[index]
        elif self.method == 'IEM':
            inputs, masks = self.data_masks_max_clip(self.inputs[index],[0])
            return inputs, masks, self.targets[index]
        elif self.method == 'IEM_pos':
            inputs, masks, pos_masks = self.data_masks_max_clip_pos(self.inputs[index],[0])
            return inputs, masks, self.targets[index], pos_masks


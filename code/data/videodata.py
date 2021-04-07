'''
    训练时，len为能取到的帧的个数
    这样训练时，能保证每一帧都能遍历到
'''

import os
import glob
import utils.utils as utils
import numpy as np
import imageio
import torch
import torch.utils.data as data
import cv2


class VIDEODATA(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.n_seq = args.n_sequences
        self.n_frames_per_video = args.n_frames_per_video
        print("n_seq:", self.n_seq)
        print("n_frames_per_video:", args.n_frames_per_video)

        self.n_frames_video = []

        if train:
            self._set_filesystem(args.dir_data)
        else:
            self._set_filesystem(args.dir_data_test)

        self.images_input = self._scan()

        self.num_video = len(self.images_input)
        #self.num_frame = sum(self.n_frames_video) - (self.n_seq - 1) * len(self.n_frames_video)
        self.num_frame = sum(self.n_frames_video)
        print("Number of videos to load:", self.num_video)
        print("Number of frames to load:", self.num_frame)

        if train:
            self.repeat = max(args.test_every // max((self.num_frame // self.args.batch_size), 1), 1)

        if args.process:
            self.data_input = self._load(self.images_input)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.dir_input = os.path.join(self.apath, 'INPUT')
        print("DataSet INPUT path:", self.dir_input)

    def _scan(self):
        vid_input_names = sorted(glob.glob(os.path.join(self.dir_input, '*')))

        images_input = []

        for vid_input_name in vid_input_names:
            if self.train:
                input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))[:self.args.n_frames_per_video]
            else:
                input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))
            images_input.append(input_dir_names)
            self.n_frames_video.append(len(input_dir_names))

        return images_input

    def _load(self, images_input):
        data_input = []

        n_videos = len(images_input)
        for idx in range(n_videos):
            if idx % 10 == 0:
                print("Loading video %d" % idx)
            inputs = np.array([imageio.imread(lr_name) for lr_name in images_input[idx]])
            data_input.append(inputs)

        return data_input

    def __getitem__(self, idx):
        if self.args.process:
            inputs, filenames = self._load_file_from_loaded_data(idx)
        else:
            inputs, filenames = self._load_file(idx)

        inputs_concat = np.zeros((inputs.shape[1], inputs.shape[2], inputs.shape[3] * inputs.shape[0]))
        for i in range(self.n_seq):
            inputs_concat[:, :, self.n_seq * i:self.n_seq * (i + 1)] = inputs[i]

        patches = self.get_patch(inputs_concat, self.args.size_must_mode)
        inputs_list = []
        input_bic_list = []

        # patch shape = H, W, 9
        for i in range(self.n_seq):
            inputs_list.append(patches[:, :, self.n_seq * i:self.n_seq * (i + 1)])
        inputs = np.array(inputs_list)
        n, h, w, c = inputs.shape
        inputs_bic_list = []
        for i in range(self.n_seq):
            inputs_bic_list.append(cv2.resize(inputs_list[i], (w * self.args.scale, h * self.args.scale), interpolation=cv2.INTER_CUBIC))
        inputs_bics = np.array(inputs_bic_list)

        inputs = np.array(utils.set_channel(*inputs, n_channels=self.args.n_colors))
        inputs_bics = np.array(utils.set_channel(*inputs_bics, n_channels=self.args.n_colors))

        input_tensors = utils.np2Tensor(*inputs, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        input_bic_tensor = utils.np2Tensor(*inputs_bics, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)

        kernel = torch.from_numpy(np.array(self.matlab_style_gauss2D())).float()

        return torch.stack(input_tensors), filenames, kernel, torch.stack(input_bic_tensor)

    def __len__(self):
        if self.train:
            return self.num_frame * 1
        else:
            return self.num_frame

    def _get_index(self, idx):
        if self.train:
            return idx % self.num_frame
        else:
            return idx

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def _load_file(self, idx):
        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)  # test时，根据idx获取对应的视频id和帧id
        f_inputs = self.images_input[video_idx][frame_idx:frame_idx + self.n_seq]
        inputs = np.array([imageio.imread(lr_name) for lr_name in f_inputs])
        filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in f_inputs]

        return inputs, filenames

    def _load_file_from_loaded_data(self, idx):
        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)  # test时，根据idx获取对应的视频id和帧id
        inputs = self.data_input[video_idx][frame_idx:frame_idx + self.n_seq]
        filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in self.inputs[video_idx][frame_idx:frame_idx + self.n_seq]]

        return inputs, filenames

    def get_patch(self, input, size_must_mode=1):
        scale = self.args.scale
        if self.train:
            input = utils.get_patch(input, patch_size=self.args.patch_size, scale=scale)
            h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input = input[:new_h, :new_w, :]
            if not self.args.no_augment:
                input = utils.data_augment(input)
        else:
            h, w, _ = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input = input[:new_h, :new_w, :]
        return input

    def matlab_style_gauss2D(self):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        shape = self.args.kernel_size
        sigma = ((shape[0] - 1) * 0.5 - 1) * 0.3 + 0.8
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

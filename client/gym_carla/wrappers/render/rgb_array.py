import copy

import gym
import numpy as np
from carla import ColorConverter, Image
from PIL import Image as PILImage


class RGBArrayRenderWrapper(gym.Wrapper):
    '''
    Basic wrapper to extract data from env RGB-available sensors
    '''

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.metadata['render.modes'] = list(set(self.metadata['render.modes'].append('rgb_array')))

    def render(self, mode='human', **kwargs):
        if mode == 'rgb_array':
            frames = []
            heights = []
            widths = []
            for k, t in self.env.vehicle_sensors_types.items():
                try:
                    converter = {
                        'sensor.camera.rgb': ColorConverter.Raw,
                        'sensor.camera.depth': ColorConverter.Depth,  # or ColorConverter.LogarithmicDepth
                        'sensor.camera.semantic_segmentation': ColorConverter.CityScapesPalette
                    }[t]
                    data: Image = copy.deepcopy(self.env.vehicle_sensors_snapshot[k])
                    data.convert(converter)

                    img = PILImage.frombuffer('RGBA', (data.width, data.height),
                                              data.raw_data, "raw", 'RGBA', 0, 1)  # load
                    img = img.convert('RGB')                                       # drop alpha
                    frames.append(img)
                    heights.append(data.height)
                    widths.append(data.width)
                except KeyError:
                    continue
            if len(frames):
                heights = np.array(heights)
                widths = np.array(widths)
                scales = heights / np.amin(heights)
                new_heights = np.full_like(heights, np.amin(heights))
                new_widths = np.round(widths / scales)
                new_frames = []
                for idx, img in enumerate(frames):
                    new_frames.append(img.resize((new_widths[idx], new_heights[idx]), resample=PILImage.BICUBIC))
                return np.concatenate(new_frames, axis=0)
        else:
            return self.env.render(mode, **kwargs)

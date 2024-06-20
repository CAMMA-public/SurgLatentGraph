from mmdet.registry import VISUALIZERS
from mmdet.visualization import DetLocalVisualizer
import os
import numpy as np
from typing import Optional
import cv2

@VISUALIZERS.register_module()
class CVSVisualizer(DetLocalVisualizer):
    def __init__(self, name: str, dataset: str = 'endoscapes',
            detector: str = 'faster_rcnn', results_dir: str = 'results',
            data_prefix: str = 'test', draw: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.save_dir = os.path.join('latent_graphs', dataset, detector)
        self.dataset = dataset

        # whether to draw detections
        self.draw = draw

        # define dir to save visualizations
        self.viz_dir = os.path.join(results_dir, '{}_preds'.format(dataset), data_prefix,
                'output_viz')

        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)

    def add_datasample(self, name: str, image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            out_file: Optional[str] = None, **kwargs):

        # extract img prefix
        if data_sample is not None:
            img_prefix = data_sample.img_path.split('/')[-1].replace('.jpg', '')
        else:
            img_prefix = name.split('/')[-1].replace('.jpg', '')

        if self.draw:
            # draw detections
            super().add_datasample(name, image, data_sample, out_file=None, **kwargs)
            viz_img = self._image

        else:
            # tile image horizontally twice
            viz_img = np.concatenate([image, image], axis=1)

        # add CVS
        self.draw_cvs(viz_img, data_sample)

        # save
        cv2.imwrite(os.path.join(self.viz_dir, img_prefix + '.jpg'),
                cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))

    def draw_cvs(self, image: np.ndarray, data_sample: Optional['DetDataSample'] = None):
        pred_c1, pred_c2, pred_c3 = data_sample.pred_ds.sigmoid().detach().cpu().tolist()
        gt_c1, gt_c2, gt_c3 = data_sample.metainfo['ds']

        # Define the text and box parameters
        pred_text = "C1: {:.2f}\nC2: {:.2f}\nC3: {:.2f}".format(pred_c1, pred_c2, pred_c3)
        gt_text = "C1: {:.2f}\nC2: {:.2f}\nC3: {:.2f}".format(gt_c1, gt_c2, gt_c3)

        # Split the text into lines
        pred_lines = pred_text.split('\n')
        gt_lines = gt_text.split('\n')

        # define fonts
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        text_color = (0, 0, 0)  # Black text

        # Get the text size to create a proper box
        line_height = cv2.getTextSize("A", font, font_scale, font_thickness)[0][1]
        box_height = line_height * len(pred_lines) + 40  # Adding some padding
        box_width = max([cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] for line in pred_lines]) + 30

        # Define the box color
        box_color = (204, 204, 0) # yellow
        alpha = 0.6  # Transparency factor

        # Calculate box position
        pred_box_coords = ((image.shape[1] - box_width - 30, image.shape[0] - box_height - 30),
                (image.shape[1] - 30, image.shape[0] - 30))
        gt_box_coords = ((image.shape[1] // 2 - box_width - 30, image.shape[0] - box_height - 30),
                (image.shape[1] // 2 - 30, image.shape[0] - 30))

        # Create the box and overlay it on the image
        overlay = image.copy()
        cv2.rectangle(overlay, pred_box_coords[0], pred_box_coords[1], box_color, -1)
        cv2.rectangle(overlay, gt_box_coords[0], gt_box_coords[1], box_color, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Calculate text position
        pred_text_x = pred_box_coords[0][0] + 10
        pred_text_y = pred_box_coords[0][1] + line_height + 10
        gt_text_x = gt_box_coords[0][0] + 10
        gt_text_y = gt_box_coords[0][1] + line_height + 10

        # Add the text on the image
        for p_line, gt_line in zip(pred_lines, gt_lines):
            cv2.putText(image, p_line, (pred_text_x, pred_text_y), font, font_scale,
                    text_color, font_thickness, lineType=cv2.LINE_AA)
            cv2.putText(image, gt_line, (gt_text_x, gt_text_y), font, font_scale,
                    text_color, font_thickness, lineType=cv2.LINE_AA)
            pred_text_y += line_height + 10
            gt_text_y += line_height + 10

        return image

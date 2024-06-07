from mmdet.registry import VISUALIZERS
from mmdet.visualization import DetLocalVisualizer
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from typing import Dict, List, Optional, Tuple, Union, Sequence
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

@VISUALIZERS.register_module()
class LatentGraphVisualizer(DetLocalVisualizer):
    def __init__(self, name: str, dataset: str = 'endoscapes',
            detector: str = 'faster_rcnn', results_dir: str = 'results',
            data_prefix: str = 'test', save_graphs: bool = False,
            gt_graph_use_pred_instances: bool = False, draw: bool = False,
            **kwargs):
        super().__init__(**kwargs)
        self.save_dir = os.path.join('latent_graphs', dataset, detector)

        self.dataset = dataset

        # whether to draw/save
        self.draw = draw
        self.save_graphs = save_graphs
        if self.save_graphs:
            # create graph dir
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        if self.draw:
            viz_dir = os.path.join(results_dir, '{}_preds'.format(dataset), data_prefix)

            self.det_viz_dir = os.path.join(viz_dir, 'detections')
            if not os.path.exists(self.det_viz_dir):
                os.makedirs(self.det_viz_dir)

            self.graph_viz_dir = os.path.join(viz_dir, 'graphs')
            if not os.path.exists(self.graph_viz_dir):
                os.makedirs(os.path.join(self.graph_viz_dir, 'gt'))
                os.makedirs(os.path.join(self.graph_viz_dir, 'pred'))

            self._init_graph_viz_info()
            # visualize gt or pred detections in graph
            self.gt_graph_use_pred_instances = gt_graph_use_pred_instances

    def _init_graph_viz_info(self):
        # Define self.colors
        cvs_datasets = ['endoscapes', 'wc', 'small_wc', 'italy', 'sages', 'sages_weighted']
        c8k_datasets = ['c80_phase', 'cholecT50']
        if self.dataset in cvs_datasets:
            self.figsize = 15
            self.colors = [
                (0, 0, 0),
                (1, 1, 0.392),
                (0.4, 0.698, 1),
                (1, 0, 0),
                (0, 0.4, 0.2),
                (0.2, 1, 0.404),
                (1, 0.592, 0.208),
                (1, 1, 1),
            ]
            self.obj_id_to_label = {
                1: 'cystic_plate',
                2: 'calot_triangle',
                3: 'cystic_artery',
                4: 'cystic_duct',
                5: 'gallbladder',
                6: 'tool'
            }
            self.obj_id_to_label_short = {
                1: 'CP',
                2: 'CT',
                3: 'CA',
                4: 'CD',
                5: 'GB',
                6: 'T',
            }

        elif self.dataset in c8k_datasets:
            self.figsize = 15
            self.colors = [
                (0, 0, 0),             # 0: Default color
                (0.753, 0.753, 0.753), # 1: Silver
                (0.541, 0.169, 0.886), # 2: Violet
                (0.502, 0.502, 0),     # 3: Olive
                (1, 1, 0.39),          # 4: Yellow
                (1, 0.592, 0.208),     # 5: Orange
                (0.4, 0.698, 1),       # 6: Light Blue
                (1, 0, 0),             # 7: Red
                (0, 0.4, 0.2),         # 8: Dark Green
                (1, 0.592, 0.208),     # 9: Orange
                (0.2, 1, 0.4),         # 10: Light Green
                (0.169, 0.216, 0.89),  # 11: Blue
                (1, 1, 1),             # 12: White
            ]

            # Define object and semantic labels
            self.obj_id_to_label = {
                1: 'abdominal_wall',
                2: 'liver',
                3: 'gastrointestinal_wall',
                4: 'fat',
                5: 'grasper',
                6: 'connective_tissue',
                7: 'blood',
                8: 'cystic_duct',
                9: 'hook',
                10: 'gallbladder',
                11: 'hepatic_vein',
                12: 'liver_ligament',
            }

            self.obj_id_to_label_short = {
                1: 'AW',
                2: 'L',
                3: 'GI',
                4: 'F',
                5: 'Gr',
                6: 'CT',
                7: 'B',
                8: 'CD',
                9: 'H',
                10: 'GB',
                11: 'HV',
                12: 'LL',
            }

        else:
            raise ValueError("Visualization not supported for this dataset (class names and self.colors not defined")

        self.sem_id_to_label = {
            1: 'LR',  # LEFT-RIGHT
            2: 'UD',  # UP-DOWN
            3: 'IO',  # INSIDE-OUTSIDE
        }

        self.sem_id_to_color = {
            1: (0.33, 0.24, 0.42),  # LEFT-RIGHT (PURPLE)
            2: (0.02, 0.77, 0.70),  # UP-DOWN (TEAL)
            3: (0.61, 0.55, 0.49),  # INSIDE-OUTSIDE (TAN)
        }

    def add_datasample(self, name: str, image: np.ndarray, pred_score_thr: float,
            data_sample: Optional['DetDataSample'] = None,
            out_file: Optional[str] = None, **kwargs):

        if self.save_graphs:
            # create filename
            graph_filename = str(data_sample.img_id) + '.npz'

            # save latent graph
            np.savez(os.path.join(self.save_dir, graph_filename), data_sample.lg.numpy())

        if self.draw:
            # extract img prefix
            if data_sample is not None:
                img_prefix = data_sample.img_path.split('/')[-1].replace('.jpg', '')
            else:
                img_prefix = name.split('/')[-1].replace('.jpg', '')

            # draw detections
            super().add_datasample(name, image, data_sample,
                    out_file=os.path.join(self.det_viz_dir, img_prefix + '.jpg'), **kwargs)

            if data_sample is not None:
                # now draw graph
                self.graphs_to_networkx(data_sample, img_prefix, pred_score_thr) # convert graphs to networkx

    def graphs_to_networkx(self, data_sample: DetDataSample, img_prefix: str,
            pred_score_thr: float):
        # Create an empty NetworkX graph for ground truth
        gt_graph = nx.Graph()

        # Get ground truth instances
        if self.gt_graph_use_pred_instances or not data_sample.is_det_keyframe:
            labels, boxes = data_sample.pred_instances.labels, data_sample.pred_instances.bboxes
        else:
            labels, boxes = data_sample.gt_instances.labels, data_sample.gt_instances.bboxes

        # Add nodes to the ground truth graph for each ground truth instance
        height, width = data_sample.ori_shape
        for i, (b, l) in enumerate(zip(boxes, labels)):
            label = self.obj_id_to_label_short[int(l.item()) + 1]

            # compute center and make sure y is from bottom left, not top right
            center = [int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)]
            center[1] = height - center[1]

            # normalize
            center[0] /= width
            center[1] /= height

            # add to graph
            gt_graph.add_node(i, label=label, color=self.colors[int(l.item()) + 1],
                    pos=center)

        # Add edges to the ground truth graph for ground truth edges
        gt_edges = data_sample.gt_edges

        for edge, rel in zip(gt_edges['edge_flats'], gt_edges['relations']):
            u, v = edge
            gt_graph.add_edge(int(u.item()), int(v.item()), color=self.sem_id_to_color[int(rel.item())],
                    relation=self.sem_id_to_label[int(rel.item())])

        # remove nodes with 0 degree (they were filtered out in gt graph gen based on score)
        gt_graph.remove_nodes_from(list(nx.isolates(gt_graph)))

        # Visualize the ground truth graph
        gt_pos = nx.get_node_attributes(gt_graph, 'pos')
        gt_node_colors = [data['color'] for _, data in gt_graph.nodes(data=True)]
        gt_edge_colors = [data['color'] for _, _, data in gt_graph.edges(data=True)]
        gt_labels = {node: data['label'] for node, data in gt_graph.nodes(data=True)}
        edge_labels = nx.get_edge_attributes(gt_graph, 'relation')
        edge_colors = [data['color'] for _, _, data in gt_graph.edges(data=True)]

        plt.figure(figsize=(self.figsize, self.figsize))
        nx.draw_networkx_nodes(gt_graph, gt_pos, node_color=gt_node_colors,
                node_size=5000, alpha=1.0)
        nx.draw_networkx_edges(gt_graph, gt_pos, edge_color=gt_edge_colors,
                width=10, alpha=0.8)

        nx.draw_networkx_labels(gt_graph, gt_pos, gt_labels, font_size=44,
                font_color='black', font_family='calibri.ttf')

        plt.axis('off')

        # Save the ground truth graph image
        plt.savefig(os.path.join(self.graph_viz_dir, 'gt', f'{img_prefix}.pdf'),
                format='pdf', transparent=True)

        # Clear the figure to free up memory
        plt.clf()
        plt.close()

        # Create an empty NetworkX graph for predicted instances
        pred_graph = nx.Graph()

        # Get pred instances (add 1 to labels to account for bg)
        pred_labels, pred_boxes = data_sample.pred_instances.labels + 1, data_sample.pred_instances.bboxes

        # Add nodes to the predicted graph for each predicted instance
        for i, (b, l) in enumerate(zip(pred_boxes, pred_labels)):
            label = self.obj_id_to_label_short[int(l.item())]

            # compute center and make sure y is from bottom left, not top right
            center = [int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)]
            center[1] = height - center[1]

            # normalize
            center[0] /= width
            center[1] /= height

            pred_graph.add_node(i, label=label, color=self.colors[int(l.item())],
                    pos=center)

        # Add edges to the predicted graph for predicted edges
        pred_edges = data_sample.pred_edges

        for edge, rel in zip(pred_edges['edge_flats'], pred_edges['relations']):
            u, v = edge
            r = rel[1:].argmax(0).int().item() + 1
            pred_graph.add_edge(int(u.item()), int(v.item()), color=self.sem_id_to_color[r],
                    relation=self.sem_id_to_label[r])

        # filter nodes based on score and remove associated edges
        inds_to_remove = data_sample.pred_instances.scores < pred_score_thr
        pred_graph.remove_nodes_from(inds_to_remove.tolist())

        # Visualize the pred graph
        pred_pos = nx.get_node_attributes(pred_graph, 'pos')
        pred_node_colors = [data['color'] for _, data in pred_graph.nodes(data=True)]
        pred_edge_colors = [data['color'] for _, _, data in pred_graph.edges(data=True)]
        pred_labels = {node: data['label'] for node, data in pred_graph.nodes(data=True)}
        edge_labels = nx.get_edge_attributes(pred_graph, 'relation')
        edge_colors = [data['color'] for _, _, data in pred_graph.edges(data=True)]

        plt.figure(figsize=(self.figsize, self.figsize))
        nx.draw_networkx_nodes(pred_graph, pred_pos, node_color=pred_node_colors,
                node_size=5000, alpha=1.0)
        nx.draw_networkx_edges(pred_graph, pred_pos, edge_color=pred_edge_colors,
                width=10, alpha=0.8)
        nx.draw_networkx_labels(pred_graph, pred_pos, pred_labels, font_size=44,
                font_color='black', font_family='calibri.ttf')
        plt.axis('off')

        # Save the pred graph
        plt.savefig(os.path.join(self.graph_viz_dir, 'pred', f'{img_prefix}.pdf'),
                format='pdf', transparent=True)

        # Clear the figure to free up memory
        plt.clf()
        plt.close()

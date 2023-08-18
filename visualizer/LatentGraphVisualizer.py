from mmdet.registry import VISUALIZERS
from mmdet.visualization import DetLocalVisualizer
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from typing import Dict, List, Optional, Tuple, Union, Sequence
import numpy as np
import torch
import os

@VISUALIZERS.register_module()
class LatentGraphVisualizer(DetLocalVisualizer):
    def __init__(self, name: str, prefix: str = 'endoscapes_faster_rcnn', save_graphs: bool = False,
            gt_graph_use_pred_instances: bool = False, draw: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix
        self.draw = draw
        self.save_graphs = save_graphs
        self.gt_graph_use_pred_instances = gt_graph_use_pred_instances

    def add_datasample(self, name: str, image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            out_file: Optional[str] = None, **kwargs):

        if self.draw:
            super().add_datasample(name, image, data_sample, out_file=out_file, **kwargs)

        if self.save_graphs:
            # create graph dir
            save_dir = os.path.join('latent_graphs', self.prefix)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # create filename
            graph_filename = str(data_sample.img_id) + '.npz'

            # save latent graph
            np.savez(os.path.join(save_dir, graph_filename), data_sample.lg.numpy())
            #torch.save(data_sample.lg, os.path.join('latent_graphs', graph_filename))
            #dump(data_sample.lg, os.path.join('latent_graphs', graph_filename))

        ## save graph object
        ##if not os.path.exists(

        ## convert graphs to networkx

        ## make dirs to save
        #if not os.path.exists(os.path.join(outfile_prefix, 'graphs')):
        #    os.makedirs(os.path.join(outfile_prefix, 'graphs', 'gt'))
        #    os.makedirs(os.path.join(outfile_prefix, 'graphs', 'pred'))

        #self.graphs_to_networkx(data_sample, [data_sample.img_id], outfile_prefix)

        #result_files['graph'] = os.path.join(outfile_prefix, 'graphs')

    def graphs_to_networkx(self, results: Sequence[dict], gts: Sequence[dict], img_ids: Sequence[int], outfile_prefix: str):
        # Define colors
        if self.prefix == 'endoscapes':
            colors = [
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
            obj_id_to_label = {
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
            obj_id_to_label_short = {
                1: 'AW',
                2: 'L',
                3: 'GI',
                4: 'F',
                5: 'Gr',
                6: 'CT',
                7: 'CD',
                8: 'H',
                9: 'GB',
                10: 'HV',
                11: 'LL',
            }
        else:
            pass

        sem_id_to_label = {
            1: 'LR',  # LEFT-RIGHT
            2: 'UD',  # UP-DOWN
            3: 'IO',  # INSIDE-OUTSIDE
        }

        sem_id_to_color = {
            1: (0.33, 0.24, 0.42),  # LEFT-RIGHT
            2: (0.02, 0.77, 0.70),  # UP-DOWN
            3: (0.61, 0.55, 0.49),  # INSIDE-OUTSIDE
        }

        # Iterate over each item in the SampleList
        for pred_item, gt_item, img_id in zip(results, gts, img_ids):
            # Create an empty NetworkX graph for ground truth
            gt_graph = nx.Graph()

            # Get ground truth instances
            if self.gt_graph_use_pred_instances:
                labels, boxes = pred_item['labels'], pred_item['bboxes']
            else:
                gt_instances = gt_item['gt_instances']
                labels, boxes = gt_instances['labels'], gt_instances['bboxes']

            # Add nodes to the ground truth graph for each ground truth instance
            for i, (b, l) in enumerate(zip(boxes, labels)):
                label = obj_id_to_label_short[int(l.item()) + 1]

                # compute center and make sure y is from bottom left, not top right
                center = [int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)]
                center[1] = gt_item['height'] - center[1]

                # normalize
                center[0] /= gt_item['width']
                center[1] /= gt_item['height']

                # add to graph
                gt_graph.add_node(i, label=label, color=colors[int(l.item()) + 1],
                        pos=center)

            # Add edges to the ground truth graph for ground truth edges
            gt_edges = gt_item['gt_edges']

            for edge, rel in zip(gt_edges['edge_flats'], gt_edges['relations']):
                u, v = edge
                gt_graph.add_edge(int(u.item()), int(v.item()), color=sem_id_to_color[int(rel.item())],
                        relation=sem_id_to_label[int(rel.item())])

            # Visualize the ground truth graph
            gt_pos = nx.get_node_attributes(gt_graph, 'pos')
            gt_node_colors = [data['color'] for _, data in gt_graph.nodes(data=True)]
            gt_edge_colors = [data['color'] for _, _, data in gt_graph.edges(data=True)]
            gt_labels = {node: data['label'] for node, data in gt_graph.nodes(data=True)}
            edge_labels = nx.get_edge_attributes(gt_graph, 'relation')
            edge_colors = [data['color'] for _, _, data in gt_graph.edges(data=True)]

            plt.figure(figsize=(10, 10))
            nx.draw_networkx_nodes(gt_graph, gt_pos, node_color=gt_node_colors,
                    node_size=5000, alpha=1.0)
            nx.draw_networkx_edges(gt_graph, gt_pos, edge_color=gt_edge_colors,
                    width=10, alpha=0.8)

            nx.draw_networkx_labels(gt_graph, gt_pos, gt_labels, font_size=44,
                    font_color='black', font_family='calibri.ttf')
            #nx.draw_networkx_edge_labels(gt_graph, gt_pos, edge_labels=edge_labels,
            #        font_size=32, font_color='black', font_family='calibri.ttf')

            plt.axis('off')

            # Save the ground truth graph image
            plt.savefig(os.path.join(outfile_prefix, 'graphs', 'gt', f'{img_id}.pdf'),
                    format='pdf', transparent=True)
            #plt.savefig(os.path.join(outfile_prefix, 'graphs', 'gt', f'{img_id}.png'),
            #        format='png', transparent=True)

            # Clear the figure to free up memory
            plt.clf()
            plt.close()

            # Create an empty NetworkX graph for predicted instances
            pred_graph = nx.Graph()

            # Get pred instances (add 1 to labels to account for bg)
            pred_labels, pred_boxes = pred_item['labels'] + 1, pred_item['bboxes']

            # Add nodes to the predicted graph for each predicted instance
            for i, (b, l) in enumerate(zip(pred_boxes, pred_labels)):
                label = obj_id_to_label_short[int(l.item())]

                # compute center and make sure y is from bottom left, not top right
                center = [int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)]
                center[1] = gt_item['height'] - center[1]

                # normalize
                center[0] /= gt_item['width']
                center[1] /= gt_item['height']

                pred_graph.add_node(i, label=label, color=colors[int(l.item())],
                        pos=center)

            # Add edges to the predicted graph for predicted edges
            pred_edges = pred_item['pred_edges']

            for edge, rel in zip(pred_edges['edge_flats'], pred_edges['relations']):
                u, v = edge
                r = rel[1:].argmax(0).int().item() + 1
                pred_graph.add_edge(int(u.item()), int(v.item()), color=sem_id_to_color[r],
                        relation=sem_id_to_label[r])

            # Visualize the ground truth graph
            pred_pos = nx.get_node_attributes(pred_graph, 'pos')
            pred_node_colors = [data['color'] for _, data in pred_graph.nodes(data=True)]
            pred_edge_colors = [data['color'] for _, _, data in pred_graph.edges(data=True)]
            pred_labels = {node: data['label'] for node, data in pred_graph.nodes(data=True)}
            edge_labels = nx.get_edge_attributes(pred_graph, 'relation')
            edge_colors = [data['color'] for _, _, data in pred_graph.edges(data=True)]

            plt.figure(figsize=(10, 10))
            nx.draw_networkx_nodes(pred_graph, pred_pos, node_color=pred_node_colors,
                    node_size=5000, alpha=1.0)
            nx.draw_networkx_edges(pred_graph, pred_pos, edge_color=pred_edge_colors,
                    width=10, alpha=0.8)
            nx.draw_networkx_labels(pred_graph, pred_pos, pred_labels, font_size=44,
                    font_color='black', font_family='calibri.ttf')
            #nx.draw_networkx_edge_labels(pred_graph, pred_pos, edge_labels=edge_labels,
            #        font_size=32, font_color='black', font_family='calibri.ttf')
            plt.axis('off')

            # Save the pred graph
            plt.savefig(os.path.join(outfile_prefix, 'graphs', 'pred', f'{img_id}.pdf'), format='pdf')

            # Clear the figure to free up memory
            plt.clf()
            plt.close()

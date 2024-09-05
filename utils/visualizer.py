import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

class Visualizer:
    @staticmethod
    def visualize_traversal(graph, traversal_path):
        traversal_graph = nx.DiGraph()
        
        for node in graph.nodes():
            traversal_graph.add_node(node)
        for u, v, data in graph.edges(data=True):
            traversal_graph.add_edge(u, v, **data)
        
        fig, ax = plt.subplots(figsize=(30, 20))  # Augmenter la taille de la figure
        
        # Utiliser un algorithme de layout qui favorise la répulsion
        pos = nx.spring_layout(traversal_graph, iterations=100, k=2, seed=42)
        for _ in range(50):  # Nombre d'itérations pour la répulsion
            for node in pos:
                dx = dy = 0
                for other in pos:
                    if node != other:
                        x1, y1 = pos[node]
                        x2, y2 = pos[other]
                        dx += (x1 - x2) / ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                        dy += (y1 - y2) / ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                pos[node] = (pos[node][0] + dx * 0.1, pos[node][1] + dy * 0.1)      
                  
        edges = traversal_graph.edges()
        edge_weights = [traversal_graph[u][v].get('weight', 0.5) for u, v in edges]
        nx.draw_networkx_edges(traversal_graph, pos, 
                               edgelist=edges,
                               edge_color=edge_weights,
                               edge_cmap=plt.cm.Blues,
                               width=1,  # Réduire la largeur des arêtes
                               ax=ax,
                               alpha=0.5)  # Rendre les arêtes plus transparentes
        
        nx.draw_networkx_nodes(traversal_graph, pos, 
                               node_color='lightblue',
                               node_size=3000,  # Réduire la taille des nœuds
                               ax=ax)
        
        edge_offset = 0.1
        for i in range(len(traversal_path) - 1):
            start = traversal_path[i]
            end = traversal_path[i + 1]
            start_pos = pos[start]
            end_pos = pos[end]
            
            mid_point = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)
            control_point = (mid_point[0] + edge_offset, mid_point[1] + edge_offset)
            
            arrow = patches.FancyArrowPatch(start_pos, end_pos,
                                            connectionstyle=f"arc3,rad={0.3}",
                                            color='red',
                                            arrowstyle="->",
                                            mutation_scale=15,  # Réduire la taille des flèches
                                            linestyle='--',
                                            linewidth=1.5,
                                            zorder=4)
            ax.add_patch(arrow)
        
        labels = {}
        for i, node in enumerate(traversal_path):
            article_title = graph.nodes[node].get('head', '')[:20]
            author = graph.nodes[node].get('author', '')
            label = f"{i + 1}. {article_title}\n({author})"
            labels[node] = label
        
        for node in traversal_graph.nodes():
            if node not in labels:
                article_title = graph.nodes[node].get('head', '')
                author = graph.nodes[node].get('author', '')
                labels[node] = f"{article_title}\n({author})"
        
        nx.draw_networkx_labels(traversal_graph, pos, labels, font_size=8, font_weight="bold", ax=ax)
        
        start_node = traversal_path[0]
        end_node = traversal_path[-1]
        
        nx.draw_networkx_nodes(traversal_graph, pos, 
                               nodelist=[start_node], 
                               node_color='lightgreen', 
                               node_size=4000,
                               ax=ax)
        
        nx.draw_networkx_nodes(traversal_graph, pos, 
                               nodelist=[end_node], 
                               node_color='lightcoral', 
                               node_size=4000,
                               ax=ax)
        
        ax.set_title("Parcours du Graphe de l'Encyclopédie", fontsize=20)
        ax.axis('off')
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Poids de l\'arête', rotation=270, labelpad=15)
        
        regular_line = plt.Line2D([0], [0], color='blue', linewidth=2, label='Arête régulière')
        traversal_line = plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Chemin de parcours')
        start_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Nœud de départ')
        end_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=15, label='Nœud d\'arrivée')
        legend = plt.legend(handles=[regular_line, traversal_line, start_point, end_point], loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
        legend.get_frame().set_alpha(0.8)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_filtered_content(traversal_path, filtered_content, graph):
        print("\nContenu filtré des nœuds visités dans l'ordre du parcours :")
        for i, node in enumerate(traversal_path):
            print(f"\nÉtape {i + 1} - Nœud {node}:")
            print(f"Article : {graph.nodes[node].get('head', 'Titre non disponible')}")
            print(f"Auteur : {graph.nodes[node].get('author', 'Auteur non disponible')}")
            print(f"Domaine : {graph.nodes[node].get('domaine_enccre', 'Domaine non disponible')}")
            print(f"Contenu filtré : {filtered_content.get(node, 'Contenu non disponible')[:200]}...")
            print("-" * 50)
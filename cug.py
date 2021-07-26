import cudf
import cugraph
M = cudf.read_csv('datasets/netscience.csv',
                      delimiter = ' ',
                      dtype=['int32', 'int32', 'float32'],
                      header=None)
G = cugraph.Graph()
G.from_cudf_edgelist(M, source='0', destination='1', edge_attr=None)
df = cugraph.weakly_connected_components(G)
print(df.shape)
print(df['labels'].max)
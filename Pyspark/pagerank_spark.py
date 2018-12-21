from operator import add
from pyspark.sql import SparkSession


def __computeContribs(urls, rank):
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    for url in urls:
        yield (url, rank / num_urls)

def pageSparkList(list, maxIterations=1000, alpha=0.15):

    # Initialize the spark context.
    spark = SparkSession \
        .builder \
        .appName("PythonPageRank") \
        .getOrCreate()

    lines = spark.sparkContext.parallelize(list)
    # Loads all URLs from input file and initialize their neighbors.
    links = lines.distinct().groupByKey().cache()

    # Loads all URLs with other URL(s) link to from input file and initialize ranks of them to one.
    ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0))

    # Calculates and updates URL ranks continuously using PageRank algorithm.
    for iteration in range(int(maxIterations)):
        # Calculates URL contributions to the rank of other URLs.
        contribs = links.join(ranks).flatMap(
            lambda url_urls_rank: __computeContribs(url_urls_rank[1][0], url_urls_rank[1][1]))

        # Re-calculates URL ranks based on neighbor contributions.
        ranks = contribs.reduceByKey(add).mapValues(lambda rank: rank * (1-alpha) + alpha)

    # Collects all URL ranks and dump them to console.
    ranks = ranks.sortBy(lambda a:a[1], ascending=False)
    # for (link, rank) in ranks.collect():
    #      print("%s has rank: %s." % (link, rank))
    res = ranks.toDF().toPandas()
    spark.stop()
    return res
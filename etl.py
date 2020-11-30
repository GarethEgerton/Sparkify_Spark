import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import TimestampType, DateType

config = configparser.ConfigParser()
config.read(os.path.expanduser("~/.aws/credentials"))

# config.read('dl.cfg')
# os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
# os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['aws_access_key_id']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['aws_secret_access_key']


def create_spark_session():
    '''
    Creates a spark session with configuration specified to be able to read from S3
    '''
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    '''
    Given a spark session, proceses raw song json files from selected input_data file location. 
    Writes songs and artists table parquet files to output file location.
    '''

    # get filepath to song data file
    song_data = input_data + 'song_data/*/*/*/*.json'
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(['song_id', 'title', 'artist_id', 'year', 'duration']).distinct()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id') \
                    .mode('overwrite') \
                    .parquet(output_data + '/songs/songs_table.parquet')

    # extract columns to create artists table
    artists_table = df.selectExpr('artist_id', 'artist_name as name', 'artist_location as location', 
                                  'artist_latitude as latitude', 'artist_longitude as longitude') \
                                      .distinct()
    
    # write artists table to parquet files
    artists_table.write.mode('overwrite') \
                        .parquet(output_data + '/artists/artists_table.parquet')

    return artists_table, songs_table


def process_log_data(spark, input_data, output_data):
    '''
    Given a spark session, proceses raw log json files from selected input_data file location. 
    Writes users, time, and songplay table parquet files to output file location.
    '''

    # get filepath to log data file
    log_data = input_data + 'log_data/*/*/*.json'

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.where(df.page == 'NextSong')

    # extract columns for users table    
    users_table = df.selectExpr('user_id', 'firstName as first_name', 
                                      'lastName as last_name', 'gender', 'level').distinct()
    
    # write users table to parquet files
    users_table.write.mode('overwrite').parquet(output_data + "/users/users_table.parquet")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x / 1000.0), TimestampType())
    df = df.withColumn('timestamp', get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_date = udf(lambda x: datetime.fromtimestamp(x / 1000.0), DateType())
    df = df.withColumn('datetime', get_date(df.ts))
    
    # extract columns to create time table
    time_table = df.selectExpr('timestamp as start_time',
                                'hour(timestamp) as hour',
                                'day(timestamp) as day',
                                'weekofyear(timestamp) as week',
                                'month(timestamp) as month',
                                'year(timestamp) as year',
                                'weekday(timestamp) as weekday').distinct()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year', 'artist_id') \
                    .mode('overwrite') \
                    .parquet(output_data + '/time/time_table.parquet')

    # read in song data to use for songplays table
    song_data = input_data + 'song_data/*/*/*/*.json'
    song_df = spark.read.json(song_data)

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = df.join(song_df, song_df.artist_name == df.artist) \
                        .join(time_table, time_table.start_time == df.timestamp) \
                        .where("page == 'NextSong'") \
                        .withColumn('songplay_id', monotonically_increasing_id()) \
                        .selectExpr('songplay_id',
                                    'start_time', 
                                    'userId as user_id',
                                    'level',
                                    'song_id',
                                    'artist_id',
                                    'sessionId as session_id',
                                    'location',
                                    'userAgent as user_agent') \
                        .sort('songplay_id')

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy('year', 'month') \
                            .mode('overwrite') \
                            .parquet(output_data + '/songs/songplays_table')

def main():
    spark = create_spark_session()
    input_data = "s3://udacity-dend/"
    output_data = "s3://gigilake/sparkify/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

if __name__ == "__main__":
    main()

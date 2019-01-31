package pagerank;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Partitioner;

public class ParsePartitioner extends Partitioner<Text, Text> {
	@Override
	public int getPartition(Text key, Text value, int numReduceTasks) {
        return key.charAt(0)%numReduceTasks;
	}
}

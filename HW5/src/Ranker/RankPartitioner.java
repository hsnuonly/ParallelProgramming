package pagerank;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Partitioner;

public class RankPartitioner extends Partitioner<Text, Text> {
	@Override
	public int getPartition(Text key, Text value, int numReduceTasks) {
        if(key.getLength()>0)
            return key.charAt(0)%numReduceTasks;
        else
            return 0;
	}
}

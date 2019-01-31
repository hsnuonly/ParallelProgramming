package pagerank;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.conf.Configuration;


public class SortReducer extends Reducer<TextPair, NullWritable, Text, Text> {
    
    long N;
    double pr_dangling = 0;

    protected void setup(Context context) throws IOException, InterruptedException{
    }

    public void reduce(TextPair key, Iterable<NullWritable> values, Context context) throws IOException, InterruptedException {
        context.write(key.getPage(),key.getRank());
    }
    
    protected void cleanup(Context context) throws IOException, InterruptedException{
    }
}

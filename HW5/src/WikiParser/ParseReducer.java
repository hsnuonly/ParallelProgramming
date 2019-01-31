package pagerank;

import java.io.IOException;
import java.util.*;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;


public class ParseReducer extends Reducer<Text, Text, Text, Text> {
    	
    
    protected void setup(Context context) throws IOException, InterruptedException {
    }

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        
        StringBuilder sb = new StringBuilder();
        for(Text t:values){
            sb.append(t.toString());
            sb.append("\t\t\t");
        }
        context.write(key,new Text(sb.toString()));
    }
    
    protected void cleanup(Context context) throws IOException, InterruptedException {
      
    }
}

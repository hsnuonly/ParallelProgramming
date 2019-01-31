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


public class ParseReducer2 extends Reducer<Text, Text, Text, Text> {
    	
    private IntWritable result = new IntWritable();
    long N = 0;
    long dangling = 0;
    MultipleOutputs mo;
    
    protected void setup(Context context) throws IOException, InterruptedException {
        mo = new MultipleOutputs(context);
    }

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        ArrayList<String> out = new ArrayList<>();
        for(Text t:values){
            String s = t.toString();
            if(s.charAt(0)=='@'){
                out.add(s.substring(1));
            }
        }
        StringBuilder sb = new StringBuilder();
        for(String s:out){
            sb.append(s);
            sb.append("\t\t\t");
        }
        sb.append(Double.toString(0));
        context.write(key,new Text(sb.toString()));
        N++;
        if(out.size()<=0)
            dangling++;
    }
    
    protected void cleanup(Context context) throws IOException, InterruptedException {
        // Configuration conf = context.getConfiguration();
        // conf.setLong("N",N);
        // context.getCounter("Custom","N").setValue(N);
        // context.getCounter("Custom","dangling").setValue(dangling);
        LongWritable lw = new LongWritable();
        lw.set(N);
        mo.write("N",lw,NullWritable.get());
        lw.set(dangling);
        mo.write("dangling",lw,NullWritable.get());
    }
}

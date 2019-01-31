package pagerank;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigDecimal;
import java.math.BigDecimal;
import java.util.StringTokenizer;


import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;


public class RankReducer extends Reducer<Text, Text, Text, Text> {
    
    long N = 0;
    double error = 0;
    double old_dangling = 0;
    double new_dangling = 0;
    MultipleOutputs mo;
    double alpha = 0.85;

    protected void setup(Context context) throws IOException, InterruptedException{
        Configuration conf = context.getConfiguration();
        N = conf.getLong("num_pages",1);
        old_dangling = Double.parseDouble(conf.get("pr_d"));
        mo = new MultipleOutputs(context);
        new_dangling = 0;
        error = 0;
    }

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        double new_rank = (0.15 + old_dangling*0.85)/N;
        double old_rank = 0;
        StringBuilder sb = new StringBuilder();
        int linkOut = 0;
        
        double inPr = 0;
        for(Text t:values){
            String str = t.toString();
            if(str.length()<=0)continue;
            else if(str.charAt(0)=='#'){
                // sb.append(str.substring(4));
                String[] ss = str.split("\t\t\t");
                linkOut = ss.length-1;
                for(int i=1;i<ss.length;i++){
                    sb.append(ss[i]);
                    sb.append("\t\t\t");
                }
            }
            else if(str.charAt(0)=='@'){
                old_rank = Double.parseDouble(str.substring(1));
            }
            else{
                inPr += Double.parseDouble(str);
            }
        }
        new_rank += inPr*0.85;
        sb.append(String.valueOf(new_rank));

        // if(key.toString()+"\t"!=sb.toString())
        context.write(key,new Text(sb.toString()));
        if(linkOut<=0)
            new_dangling += new_rank;
        error += Math.abs(new_rank-old_rank);

    }
    
    protected void cleanup(Context context) throws IOException, InterruptedException{
        
        DoubleWritable dw = new DoubleWritable();
        dw.set(error);
        mo.write("error",dw,NullWritable.get());
        dw.set(new_dangling);
        mo.write("dangling",dw,NullWritable.get());
    }
}

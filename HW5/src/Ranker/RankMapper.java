package pagerank;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.StringTokenizer;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.conf.Configuration;

import java.util.ArrayList;
import java.util.Arrays;
import java.net.URI;
import java.io.*;

public class RankMapper extends Mapper<Text, Text, Text, Text> {
    Text k = new Text();
    Text v = new Text();
    long N;
    Double pr_dangling;
    int iter = 0;

    protected void setup(Context context) throws IOException, InterruptedException{
        Configuration conf = context.getConfiguration();
        N = conf.getLong("num_pages",1);
        pr_dangling = Double.parseDouble(conf.get("pr_d"));
        iter = Integer.valueOf(conf.get("iter"));
    }

    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        String[] s = value.toString().split("\t\t\t");
        // Text page = new Text(s[0]);
        double rank = Double.parseDouble(s[s.length-1]);
        if(iter<=0){
            rank = 1.0/(double)N;
        } 
        int linkOut = 0;
                
        // Links
        StringBuilder sb = new StringBuilder();
        sb.append("#");
        sb.append("\t\t\t");
        for(int i=0;i<s.length-1;i++){
            sb.append(s[i]);
            sb.append("\t\t\t");
            if(s[i].length()>0)linkOut++;
        }
        context.write(key,new Text(sb.toString()));

        
        // Old Rank Pass
        context.write(key,new Text("@"+String.valueOf(rank)));

        // Rank Mapping
        if(linkOut<=0)return;
        double rank_per_link = rank/linkOut;
        v.set(String.valueOf(rank_per_link));
        for(int i=0;i<s.length-1;i++){
            if(s[i].length()<=0)continue;
            k.set(s[i]);
            context.write(k,v);
        }
    }
}

package pagerank;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
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

public class SortMapper extends Mapper<Text, Text, TextPair, NullWritable> {
    Text k = new Text();
    Text v = new Text();
    // long N;
    // Double pr_dangling;

    protected void setup(Context context) throws IOException, InterruptedException{
        Configuration conf = context.getConfiguration();
    }

    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        String[] s = value.toString().split("\t\t\t");
        // Text page = new Text(s[0]);
        // double rank = Double.valueOf(s[s.length-1]);
        if(s.length<1)return;
        k.set(key);
        v.set(s[s.length-1]);
        context.write(new TextPair(k,v),NullWritable.get());
    }
}

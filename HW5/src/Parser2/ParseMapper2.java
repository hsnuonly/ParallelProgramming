package pagerank;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;

import java.util.ArrayList;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.conf.Configuration;

import java.net.URI;
import java.io.*;

public class ParseMapper2 extends Mapper<Text, Text, Text, Text> {

    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        String[] ss = value.toString().split("\t\t\t");
        String keyString = key.toString();
        boolean existing = false;
        
        ArrayList<String> out = new ArrayList<>();
        ArrayList<String> in = new ArrayList<>();
        for(String s : ss){
            if(s.length()>0)
            switch (s.charAt(0)) {
                // out from:to
                // case '#':
                //     out.add(s);
                //     break;
                // in to:from
                case '@':
                    in.add(s);
                    break;
                // existing
                case '!':
                    existing = true;
                    break;
                default:
                    break;
            }
        }
        Text atKey = new Text("@"+keyString);
        if(existing){
            for(String inLink:in){
                context.write(new Text(inLink.substring(1)),atKey);
            }
            // for(String outLink:out){
            //     context.write(key,new Text(outLink));
            // }
            context.write(key,new Text("!"));
        }
    }
}

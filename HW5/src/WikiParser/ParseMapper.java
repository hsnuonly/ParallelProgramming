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

public class ParseMapper extends Mapper<LongWritable, Text, Text, Text> {

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

        /* Match title pattern */
        Pattern titlePattern = Pattern.compile("<title>(.+?)</title>");
        // Matcher titleMatcher = titlePattern.matcher(xxx);
        String processed = unescapeXML(value.toString());
        Matcher titleMatcher = titlePattern.matcher(processed);
        // No need capitalizeFirstLetter
        String title;
        if(titleMatcher.find()){
            title = titleMatcher.group(1);
        }
        else return;

        /* Match link pattern */
        Pattern linkPattern = Pattern.compile("\\[\\[(.+?)([\\|#]|\\]\\])");
        Matcher linkMatcher = linkPattern.matcher(processed);
        
        while(linkMatcher.find()){
            String link = linkMatcher.group(1);
            link = capitalizeFirstLetter(link);
            // out
            // context.write(new Text(title),new Text("#"+link));
            // in
            context.write(new Text(link),new Text("@"+title));
        }
        context.write(new Text(title),new Text("!"));
    }

    private String unescapeXML(String input) {

        return input.replaceAll("&lt;", "<").replaceAll("&gt;", ">").replaceAll("&amp;", "&").replaceAll("&quot;", "\"")
                .replaceAll("&apos;", "\'");

    }

    private String capitalizeFirstLetter(String input) {

        char firstChar = input.charAt(0);

        if (firstChar >= 'a' && firstChar <= 'z') {
            if (input.length() == 1) {
                return input.toUpperCase();
            } else
                return input.substring(0, 1).toUpperCase() + input.substring(1);
        } else
            return input;
    }
}

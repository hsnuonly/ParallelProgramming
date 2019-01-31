package pagerank;

import java.io.BufferedReader;
import java.io.InputStreamReader;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.fs.FSDataInputStream;

import pagerank.ParseMapper;

public class WikiParser {
    private PageRank parent;
    public WikiParser(PageRank p){
        this.parent = p;
    }

	public void run(String input,String output) throws Exception {
		Configuration conf = new Configuration();
		
		Job job = Job.getInstance(conf, "WikiParser");
		job.setJarByClass(WikiParser.class);
        conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator", "\t\t");
        conf.set(TextOutputFormat.SEPERATOR, "\t\t");
		
		// set the class of each stage in mapreduce
		job.setMapperClass(ParseMapper.class);
		job.setPartitionerClass(ParsePartitioner.class);
		// job.setSortComparatorClass(ParseComparator.class);
        job.setReducerClass(ParseReducer.class);
        
		
		// set the output class of Mapper and Reducer
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		
		// set the number of reducer
        job.setNumReduceTasks(this.parent.partition);
        
        Path inputPath = new Path(input);
        Path tmpPath = new Path(output,"tmp");
        FileSystem fs = inputPath.getFileSystem(conf);
        fs.delete(tmpPath,true);
        
		// add input/output path
		FileInputFormat.addInputPath(job, inputPath);
		FileOutputFormat.setOutputPath(job, tmpPath);
        MultipleOutputs.addNamedOutput(job, "dangling", TextOutputFormat.class,
        LongWritable.class, NullWritable.class);
        MultipleOutputs.addNamedOutput(job, "N", TextOutputFormat.class,
        LongWritable.class, NullWritable.class);
		
        job.waitForCompletion(true);
    }
    
    
}

package pagerank;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;

import pagerank.ParseMapper;

public class Sorter {
    PageRank parent;
    public Sorter(PageRank parent){
        this.parent = parent;
    }

	public void run(String path,int iter) throws Exception {
		Configuration conf = new Configuration();
        conf.set("num_pages",String.valueOf(parent.N));
        conf.set("pr_d",String.valueOf(parent.pr_dangling));
		
        Job job = Job.getInstance(conf, "Sorter");
        job.setInputFormatClass(KeyValueTextInputFormat.class);	
        conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator", "\t\t");
		
		job.setJarByClass(Sorter.class);
		
		// set the class of each stage in mapreduce
		job.setMapperClass(SortMapper.class);
		// job.setPartitionerClass(SortPartitioner.class);
		// job.setSortComparatorClass(SortComparator.class);
        job.setReducerClass(SortReducer.class);
        
		
		// set the output class of Mapper and Reducer
		job.setMapOutputKeyClass(TextPair.class);
		job.setMapOutputValueClass(NullWritable.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		
		// set the number of reducer
        // job.setNumReduceTasks(4);
        Path dir = new Path(path);
        Path inputPath = new Path(dir,"tmp"+iter);
        Path outputPath = new Path(dir,"result");
        FileSystem fs = inputPath.getFileSystem(conf);
        fs.delete(outputPath,true);
        
		// add input/output path
		FileInputFormat.addInputPath(job, inputPath);
		FileOutputFormat.setOutputPath(job, outputPath);
		
		job.waitForCompletion(true);
        fs.delete(inputPath,true);
    }
    
    
}

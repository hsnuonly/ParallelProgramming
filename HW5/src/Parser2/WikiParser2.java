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
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.fs.FSDataInputStream;

import pagerank.ParseMapper;

public class WikiParser2 {
    private PageRank parent;
    public WikiParser2(PageRank p){
        this.parent = p;
    }

	public void run(String path) throws Exception {
		Configuration conf = new Configuration();
		
		Job job = Job.getInstance(conf, "WikiParser");
        job.setJarByClass(WikiParser2.class);
        job.setInputFormatClass(KeyValueTextInputFormat.class);	
        conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator", "\t\t");
        conf.set(TextOutputFormat.SEPERATOR, "\t\t");
		
		// set the class of each stage in mapreduce
		job.setMapperClass(ParseMapper2.class);
		job.setPartitionerClass(ParsePartitioner2.class);
		// job.setSortComparatorClass(ParseComparator2.class);
        job.setReducerClass(ParseReducer2.class);
        
		
		// set the output class of Mapper and Reducer
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		
		// set the number of reducer
        job.setNumReduceTasks(this.parent.partition);
        
        Path inputPath = new Path(path,"tmp");
        Path tmpPath = new Path(path,"tmp0");
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
        
        long dangling = 0;
        for(int i=0;;i++){
            Path danglingPath = new Path(tmpPath,"dangling"+"-r-"+String.format("%05d",i));
            if(!fs.exists(danglingPath))break;
            FSDataInputStream fdsis = fs.open(danglingPath);
            BufferedReader br = new BufferedReader(new InputStreamReader(fdsis));
            while(true){
                String line = br.readLine();
                if(line==null)break;
                dangling+=Long.valueOf(line);
            }
            br.close();
            fs.delete(danglingPath,true);
        }

        
        long N = 0;
        for(int i=0;;i++){
            Path NPath = new Path(tmpPath,"N"+"-r-"+String.format("%05d",i));
            if(!fs.exists(NPath))break;
            FSDataInputStream fdsis = fs.open(NPath);
            BufferedReader br = new BufferedReader(new InputStreamReader(fdsis));
            while(true){
                String line = br.readLine();
                if(line==null)break;
                N+=Long.valueOf(line);
            }
            br.close();
            fs.delete(NPath,true);
        }

        // return job.getCounters().findCounter("Custom","N").getValue();
        // this.parent.setN(job.getCounters().findCounter("Custom","N").getValue());
        // double pr_d = (double)job.getCounters().findCounter("Custom","N").getValue()/
                    // (double)job.getCounters().findCounter("Custom","dangling").getValue();
        this.parent.setPR_dangling((double)dangling/N);
        this.parent.setN(N);
        // return conf.getLong("N",2);
    }
    
    
}

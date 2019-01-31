package pagerank;

import java.io.BufferedReader;
import java.io.InputStreamReader;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.fs.FSDataInputStream;

import pagerank.ParseMapper;

public class Ranker {
    PageRank parent;
    public Ranker(PageRank parent){
        this.parent = parent;
    }

	public double run(String path,int iter) throws Exception {
		Configuration conf = new Configuration();
        conf.set("num_pages",String.valueOf(parent.N));
        conf.set("pr_d",String.valueOf(parent.pr_dangling));
        conf.set("iter",String.valueOf(iter));
		
        Job job = Job.getInstance(conf, "Ranker");
        job.setInputFormatClass(KeyValueTextInputFormat.class);	
        conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator", "\t\t");
        conf.set(TextOutputFormat.SEPERATOR, "\t\t");

		job.setJarByClass(Ranker.class);
		
		// set the class of each stage in mapreduce
		job.setMapperClass(RankMapper.class);
		job.setPartitionerClass(RankPartitioner.class);
		// job.setSortComparatorClass(RankComparator.class);
        job.setReducerClass(RankReducer.class);
        
		
		// set the output class of Mapper and Reducer
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		
		// set the number of reducer
        job.setNumReduceTasks(this.parent.partition);
        Path dir = new Path(path);
        Path inputPath = new Path(dir,"tmp"+iter);
        Path outputPath = new Path(dir,"tmp"+(iter+1));
        FileSystem fs = inputPath.getFileSystem(conf);
        fs.delete(outputPath,true);

        MultipleOutputs.addNamedOutput(job, "dangling", TextOutputFormat.class,
        DoubleWritable.class, NullWritable.class);
        MultipleOutputs.addNamedOutput(job, "error", TextOutputFormat.class,
        DoubleWritable.class, NullWritable.class);
        
		// add input/output path
		FileInputFormat.addInputPath(job, inputPath);
		FileOutputFormat.setOutputPath(job, outputPath);
        

        job.waitForCompletion(true);
        
        double dangling = 0;
        for(int i=0;;i++){
            Path danglingPath = new Path(outputPath,"dangling"+"-r-"+String.format("%05d",i));
            if(!fs.exists(danglingPath))break;
            FSDataInputStream fdsis = fs.open(danglingPath);
            BufferedReader br = new BufferedReader(new InputStreamReader(fdsis));
            while(true){
                String line = br.readLine();
                if(line==null)break;
                dangling+=Double.valueOf(line);
            }
            br.close();
            fs.delete(danglingPath,true);
        }
        
        double error = 0;
        for(int i=0;;i++){
            Path errorPath = new Path(outputPath,"error"+"-r-"+String.format("%05d",i));
            if(!fs.exists(errorPath))break;
            FSDataInputStream fdsis = fs.open(errorPath);
            BufferedReader br = new BufferedReader(new InputStreamReader(fdsis));
            while(true){
                String line = br.readLine();
                if(line==null)break;
                error+=Double.valueOf(line);
            }
            br.close();
            fs.delete(errorPath,true);
        }
        fs.delete(inputPath,true);
        
        parent.setPR_dangling(dangling);
        return error;
    }
    
    
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <float.h>
#include "mpi.h"


int main(int argc,char** argv)
{
    int rank, size;
    float* buf;
    int sorted;
    if(argc<4){
        return 1;
    }

    // initialize mpi
    int rc = MPI_Init(&argc, &argv);

    // error handling for mpi
    if(rc!= MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    // initialize rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int slices = atoi(argv[1]);
    int slices_per_process = slices / size;

    MPI_Comm comm = MPI_COMM_WORLD;

    if(slices_per_process<=0){
        slices_per_process = 1;
        size = slices;
        MPI_Group orig_group, new_group;
        MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
        int ranks1[slices],ranks2[size-slices];
        for(int i=0;i<slices;i++)ranks1[i] = i;
        for(int i=0;i<size-slices;i++)ranks2[i]=i+slices;
        if(rank<slices){
            MPI_Group_incl(orig_group,slices,ranks1,&new_group);
            MPI_Comm_create(MPI_COMM_WORLD, new_group, &comm);
        }
        else{
            MPI_Group_incl(orig_group,slices,ranks2,&new_group);
            MPI_Comm_create(MPI_COMM_WORLD, new_group, &comm);
            MPI_Finalize();
            return 0;
        }
    }
    
    if(rank==size-1 && slices_per_process*size<slices)
        slices_per_process = slices-slices_per_process*(size-1);

    buf = new float[slices_per_process];
    for(int i=0;i<slices_per_process;i++)buf[i]=FLT_MAX;
    
    MPI_File input_file;
    MPI_Status status;

    MPI_File_open(comm,argv[2],MPI_MODE_RDONLY,MPI_INFO_NULL,
        &input_file);

    MPI_File_read_ordered(input_file,buf,slices_per_process,MPI_FLOAT,&status);

    MPI_File_close(&input_file);


    MPI_Barrier(comm);
    sorted = 0;
    for(int j=0;;j++){
        //for(int i=0;i<slices_per_process;i++)printf("%f\n",buf[i]);
        if(j%2)sorted = 1;
        for(int i=0;i+1<slices_per_process;i+=2){
            if(buf[i]>buf[i+1]){
                std::swap(buf[i],buf[i+1]);
                sorted = 0;
            }
        }
        for(int i=1;i+1<slices_per_process;i+=2){
            if(buf[i]>buf[i+1]){
                std::swap(buf[i],buf[i+1]);
                sorted = 0;
            }
        }
        MPI_Barrier(comm);
        float left,right;
        MPI_Request req;
        if(slices/size==1){
            if(rank%2==j%2 && rank+1<size)MPI_Isend(&buf[slices_per_process-1],1,MPI_FLOAT,rank+1,0,comm,&req);
            if(rank%2!=j%2 && rank>0)MPI_Isend(&buf[0],1,MPI_FLOAT,rank-1,0,comm,&req);

            if(rank%2==j%2 && rank+1<size){
                MPI_Recv(&right,1,MPI_FLOAT,rank+1,0,comm,&status);
                if(right<buf[slices_per_process-1]){
                    buf[slices_per_process-1]=right;
                    sorted = 0;
                }
            }
            if(rank%2!=j%2 && rank>0){
                MPI_Recv(&left,1,MPI_FLOAT,rank-1,0,comm,&status);
                if(left>buf[0]){
                    buf[0]=left;
                    sorted = 0;
                }
            }
        }
        else{
            if(rank+1<size)MPI_Isend(&buf[slices_per_process-1],1,MPI_FLOAT,rank+1,0,comm,&req);
            if(rank>0)MPI_Isend(&buf[0],1,MPI_FLOAT,rank-1,0,comm,&req);
            if(rank+1<size){
                MPI_Recv(&right,1,MPI_FLOAT,rank+1,0,comm,&status);
                if(right<buf[slices_per_process-1]){
                    buf[slices_per_process-1]=right;
                    sorted = 0;
                }
            }
            if(rank>0){
                MPI_Recv(&left,1,MPI_FLOAT,rank-1,0,comm,&status);
                if(left>buf[0]){
                    buf[0]=left;
                    sorted = 0;
                }
            }
        }
        MPI_Barrier(comm);

        int total_sorted = 0;
        MPI_Allreduce(&sorted,&total_sorted,1,MPI_INT,MPI_LAND,comm);
        if(total_sorted)break;
    }


    MPI_Barrier(comm);
    MPI_File output_file;

    MPI_File_open(comm,argv[3],MPI_MODE_CREATE|MPI_MODE_RDWR,MPI_INFO_NULL,
        &output_file);
    /*
    if(rank==size-1&&rank!=0)
        MPI_File_seek(output_file,slices-slices_per_process*sizeof(float),MPI_SEEK_SET);
    else
        MPI_File_seek(output_file,slices_per_process*rank*sizeof(float),MPI_SEEK_SET);
    */
    //MPI_File_write(output_file,buf,slices_per_process,MPI_FLOAT,&status);

    MPI_File_write_ordered(output_file,buf,slices_per_process,MPI_FLOAT,&status);
    
    MPI_File_close(&output_file);

    END_BEFORE_FANALIZE:
    MPI_Finalize();
    return 0;
}


//
// name : Ciro Ceissler
// email: ciro.ceissler@gmail.com
// RA   : RA108786
//
// description: T2
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void producer_consumer(int size, int *vec, int n, int nt) {
	int i, j;
	long long unsigned int sum = 0;

// NOTES(ciroceissler): criar cada thread uma unica vez.
# pragma omp parallel num_threads(nt) default(none) \
  private (i, j) shared(vec, size, n) reduction(+: sum)
  {
    int *buffer_thread = (int *) malloc(size*sizeof(int));

// NOTES(ciroceissler): garantir que cada thread seja produtor/consumidor
# pragma omp for schedule(static, 2)
    for(i=0;i<n;i++) {
      if(i % 2 == 0) {	// PRODUTOR
        for(j=0;j<size;j++) {
          buffer_thread[j] = vec[i] + j*vec[i+1];
        }
      }
      else {	// CONSUMIDOR
        for(j=0;j<size;j++) {
          sum += buffer_thread[j];
        }
      }
    }

    free(buffer_thread);
  }

	printf("%llu\n",sum);
}

int main(int argc, char * argv[]) {
	double start, end;
	int i, n, size, nt;
	int *vec;

	scanf("%d %d %d",&nt,&n,&size);

	vec = (int *)malloc(n*sizeof(int));

	for(i=0;i<n;i++)
		scanf("%d",&vec[i]);

	start = omp_get_wtime();
	producer_consumer(size, vec, n, nt);
	end = omp_get_wtime();

	printf("%lf\n",end-start);

	free(vec);

	return 0;
}

/////////////////////////////////////
// cpuinfo
/////////////////////////////////////

// $ cat /proc/cpuinfo
//
// processor	: 0
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 58
// model name	: Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz
// stepping	: 9
// cpu MHz		: 1600.000
// cache size	: 8192 KB
// physical id	: 0
// siblings	: 8
// core id		: 0
// cpu cores	: 4
// apicid		: 0
// initial apicid	: 0
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm ida arat epb xsaveopt pln pts dts tpr_shadow vnmi flexpriority ept vpid fsgsbase smep erms
// bogomips	: 6784.24
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 36 bits physical, 48 bits virtual
// power management:
//
// processor	: 1
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 58
// model name	: Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz
// stepping	: 9
// cpu MHz		: 2800.000
// cache size	: 8192 KB
// physical id	: 0
// siblings	: 8
// core id		: 1
// cpu cores	: 4
// apicid		: 2
// initial apicid	: 2
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm ida arat epb xsaveopt pln pts dts tpr_shadow vnmi flexpriority ept vpid fsgsbase smep erms
// bogomips	: 6784.24
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 36 bits physical, 48 bits virtual
// power management:
//
// processor	: 2
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 58
// model name	: Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz
// stepping	: 9
// cpu MHz		: 1600.000
// cache size	: 8192 KB
// physical id	: 0
// siblings	: 8
// core id		: 2
// cpu cores	: 4
// apicid		: 4
// initial apicid	: 4
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm ida arat epb xsaveopt pln pts dts tpr_shadow vnmi flexpriority ept vpid fsgsbase smep erms
// bogomips	: 6784.24
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 36 bits physical, 48 bits virtual
// power management:
//
// processor	: 3
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 58
// model name	: Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz
// stepping	: 9
// cpu MHz		: 1900.000
// cache size	: 8192 KB
// physical id	: 0
// siblings	: 8
// core id		: 3
// cpu cores	: 4
// apicid		: 6
// initial apicid	: 6
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm ida arat epb xsaveopt pln pts dts tpr_shadow vnmi flexpriority ept vpid fsgsbase smep erms
// bogomips	: 6784.24
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 36 bits physical, 48 bits virtual
// power management:
//
// processor	: 4
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 58
// model name	: Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz
// stepping	: 9
// cpu MHz		: 1600.000
// cache size	: 8192 KB
// physical id	: 0
// siblings	: 8
// core id		: 0
// cpu cores	: 4
// apicid		: 1
// initial apicid	: 1
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm ida arat epb xsaveopt pln pts dts tpr_shadow vnmi flexpriority ept vpid fsgsbase smep erms
// bogomips	: 6784.24
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 36 bits physical, 48 bits virtual
// power management:
//
// processor	: 5
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 58
// model name	: Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz
// stepping	: 9
// cpu MHz		: 1600.000
// cache size	: 8192 KB
// physical id	: 0
// siblings	: 8
// core id		: 1
// cpu cores	: 4
// apicid		: 3
// initial apicid	: 3
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm ida arat epb xsaveopt pln pts dts tpr_shadow vnmi flexpriority ept vpid fsgsbase smep erms
// bogomips	: 6784.24
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 36 bits physical, 48 bits virtual
// power management:
//
// processor	: 6
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 58
// model name	: Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz
// stepping	: 9
// cpu MHz		: 1600.000
// cache size	: 8192 KB
// physical id	: 0
// siblings	: 8
// core id		: 2
// cpu cores	: 4
// apicid		: 5
// initial apicid	: 5
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm ida arat epb xsaveopt pln pts dts tpr_shadow vnmi flexpriority ept vpid fsgsbase smep erms
// bogomips	: 6784.24
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 36 bits physical, 48 bits virtual
// power management:
//
// processor	: 7
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 58
// model name	: Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz
// stepping	: 9
// cpu MHz		: 1600.000
// cache size	: 8192 KB
// physical id	: 0
// siblings	: 8
// core id		: 3
// cpu cores	: 4
// apicid		: 7
// initial apicid	: 7
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm ida arat epb xsaveopt pln pts dts tpr_shadow vnmi flexpriority ept vpid fsgsbase smep erms
// bogomips	: 6784.24
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 36 bits physical, 48 bits virtual
// power management:

/////////////////////////////////////
// profile
/////////////////////////////////////

// $ gprof -p -b T2_ser < arq3.in
//
// Flat profile:
//
// Each sample counts as 0.01 seconds.
//   %   cumulative   self              self     total
//  time   seconds   seconds    calls   s/call   s/call  name
// 100.75      2.34     2.34        1     2.34     2.34  producer_consumer

/////////////////////////////////////
// uso da flag On
/////////////////////////////////////

// versao serial:
//
// 01: 0.483115
// 02: 0.446782
// 03: 0.380808

// versao paralela:
//
// O1: 0.299279
// O2: 0.247299
// O3: 0.190284

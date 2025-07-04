// Test images data - hosted on Cloudinary for production
// Total images available: 110

export interface TestImage {
  id: number;
  name: string;
  description: string;
  url: string;
  category: string;
  difficulty: string;
  expectedResult: string;
}

export const ALL_TEST_IMAGES: TestImage[] = [
  {
    "id": 1,
    "name": "Test Image 001",
    "description": "SAR satellite image - Oil spill detection test case 1",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637884/oil-spill-test-images/img_0001.jpg",
    "category": "coastal",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 2,
    "name": "Test Image 002",
    "description": "SAR satellite image - Oil spill detection test case 2",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637886/oil-spill-test-images/img_0002.jpg",
    "category": "offshore",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 3,
    "name": "Test Image 003",
    "description": "SAR satellite image - Oil spill detection test case 3",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637887/oil-spill-test-images/img_0003.jpg",
    "category": "complex",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 4,
    "name": "Test Image 004",
    "description": "SAR satellite image - Oil spill detection test case 4",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637889/oil-spill-test-images/img_0004.jpg",
    "category": "satellite",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 5,
    "name": "Test Image 005",
    "description": "SAR satellite image - Oil spill detection test case 5",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637890/oil-spill-test-images/img_0005.jpg",
    "category": "coastal",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 6,
    "name": "Test Image 006",
    "description": "SAR satellite image - Oil spill detection test case 6",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637892/oil-spill-test-images/img_0006.jpg",
    "category": "offshore",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 7,
    "name": "Test Image 007",
    "description": "SAR satellite image - Oil spill detection test case 7",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637894/oil-spill-test-images/img_0007.jpg",
    "category": "complex",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 8,
    "name": "Test Image 008",
    "description": "SAR satellite image - Oil spill detection test case 8",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637895/oil-spill-test-images/img_0008.jpg",
    "category": "satellite",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 9,
    "name": "Test Image 009",
    "description": "SAR satellite image - Oil spill detection test case 9",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637897/oil-spill-test-images/img_0009.jpg",
    "category": "coastal",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 10,
    "name": "Test Image 010",
    "description": "SAR satellite image - Oil spill detection test case 10",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637898/oil-spill-test-images/img_0010.jpg",
    "category": "offshore",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 11,
    "name": "Test Image 011",
    "description": "SAR satellite image - Oil spill detection test case 11",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637900/oil-spill-test-images/img_0011.jpg",
    "category": "complex",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 12,
    "name": "Test Image 012",
    "description": "SAR satellite image - Oil spill detection test case 12",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637902/oil-spill-test-images/img_0012.jpg",
    "category": "satellite",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 13,
    "name": "Test Image 013",
    "description": "SAR satellite image - Oil spill detection test case 13",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637904/oil-spill-test-images/img_0013.jpg",
    "category": "coastal",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 14,
    "name": "Test Image 014",
    "description": "SAR satellite image - Oil spill detection test case 14",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637907/oil-spill-test-images/img_0014.jpg",
    "category": "offshore",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 15,
    "name": "Test Image 015",
    "description": "SAR satellite image - Oil spill detection test case 15",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637908/oil-spill-test-images/img_0015.jpg",
    "category": "complex",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 16,
    "name": "Test Image 016",
    "description": "SAR satellite image - Oil spill detection test case 16",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637910/oil-spill-test-images/img_0016.jpg",
    "category": "satellite",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 17,
    "name": "Test Image 017",
    "description": "SAR satellite image - Oil spill detection test case 17",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637912/oil-spill-test-images/img_0017.jpg",
    "category": "coastal",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 18,
    "name": "Test Image 018",
    "description": "SAR satellite image - Oil spill detection test case 18",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637913/oil-spill-test-images/img_0018.jpg",
    "category": "offshore",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 19,
    "name": "Test Image 019",
    "description": "SAR satellite image - Oil spill detection test case 19",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637915/oil-spill-test-images/img_0019.jpg",
    "category": "complex",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 20,
    "name": "Test Image 020",
    "description": "SAR satellite image - Oil spill detection test case 20",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637917/oil-spill-test-images/img_0020.jpg",
    "category": "satellite",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 21,
    "name": "Test Image 021",
    "description": "SAR satellite image - Oil spill detection test case 21",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637918/oil-spill-test-images/img_0021.jpg",
    "category": "coastal",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 22,
    "name": "Test Image 022",
    "description": "SAR satellite image - Oil spill detection test case 22",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637920/oil-spill-test-images/img_0022.jpg",
    "category": "offshore",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 23,
    "name": "Test Image 023",
    "description": "SAR satellite image - Oil spill detection test case 23",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637922/oil-spill-test-images/img_0023.jpg",
    "category": "complex",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 24,
    "name": "Test Image 024",
    "description": "SAR satellite image - Oil spill detection test case 24",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637924/oil-spill-test-images/img_0024.jpg",
    "category": "satellite",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 25,
    "name": "Test Image 025",
    "description": "SAR satellite image - Oil spill detection test case 25",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637927/oil-spill-test-images/img_0025.jpg",
    "category": "coastal",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 26,
    "name": "Test Image 026",
    "description": "SAR satellite image - Oil spill detection test case 26",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637929/oil-spill-test-images/img_0026.jpg",
    "category": "offshore",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 27,
    "name": "Test Image 027",
    "description": "SAR satellite image - Oil spill detection test case 27",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637930/oil-spill-test-images/img_0027.jpg",
    "category": "complex",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 28,
    "name": "Test Image 028",
    "description": "SAR satellite image - Oil spill detection test case 28",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637932/oil-spill-test-images/img_0028.jpg",
    "category": "satellite",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 29,
    "name": "Test Image 029",
    "description": "SAR satellite image - Oil spill detection test case 29",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637934/oil-spill-test-images/img_0029.jpg",
    "category": "coastal",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 30,
    "name": "Test Image 030",
    "description": "SAR satellite image - Oil spill detection test case 30",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637935/oil-spill-test-images/img_0030.jpg",
    "category": "offshore",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 31,
    "name": "Test Image 031",
    "description": "SAR satellite image - Oil spill detection test case 31",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637937/oil-spill-test-images/img_0031.jpg",
    "category": "complex",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 32,
    "name": "Test Image 032",
    "description": "SAR satellite image - Oil spill detection test case 32",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637939/oil-spill-test-images/img_0032.jpg",
    "category": "satellite",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 33,
    "name": "Test Image 033",
    "description": "SAR satellite image - Oil spill detection test case 33",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637941/oil-spill-test-images/img_0033.jpg",
    "category": "coastal",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 34,
    "name": "Test Image 034",
    "description": "SAR satellite image - Oil spill detection test case 34",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637944/oil-spill-test-images/img_0034.jpg",
    "category": "offshore",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 35,
    "name": "Test Image 035",
    "description": "SAR satellite image - Oil spill detection test case 35",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637946/oil-spill-test-images/img_0035.jpg",
    "category": "complex",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 36,
    "name": "Test Image 036",
    "description": "SAR satellite image - Oil spill detection test case 36",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637948/oil-spill-test-images/img_0036.jpg",
    "category": "satellite",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 37,
    "name": "Test Image 037",
    "description": "SAR satellite image - Oil spill detection test case 37",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637949/oil-spill-test-images/img_0037.jpg",
    "category": "coastal",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 38,
    "name": "Test Image 038",
    "description": "SAR satellite image - Oil spill detection test case 38",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637951/oil-spill-test-images/img_0038.jpg",
    "category": "offshore",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 39,
    "name": "Test Image 039",
    "description": "SAR satellite image - Oil spill detection test case 39",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637952/oil-spill-test-images/img_0039.jpg",
    "category": "complex",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 40,
    "name": "Test Image 040",
    "description": "SAR satellite image - Oil spill detection test case 40",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637954/oil-spill-test-images/img_0040.jpg",
    "category": "satellite",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 41,
    "name": "Test Image 041",
    "description": "SAR satellite image - Oil spill detection test case 41",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637955/oil-spill-test-images/img_0041.jpg",
    "category": "coastal",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 42,
    "name": "Test Image 042",
    "description": "SAR satellite image - Oil spill detection test case 42",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637956/oil-spill-test-images/img_0042.jpg",
    "category": "offshore",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 43,
    "name": "Test Image 043",
    "description": "SAR satellite image - Oil spill detection test case 43",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637958/oil-spill-test-images/img_0043.jpg",
    "category": "complex",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 44,
    "name": "Test Image 044",
    "description": "SAR satellite image - Oil spill detection test case 44",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637960/oil-spill-test-images/img_0044.jpg",
    "category": "satellite",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 45,
    "name": "Test Image 045",
    "description": "SAR satellite image - Oil spill detection test case 45",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637962/oil-spill-test-images/img_0045.jpg",
    "category": "coastal",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 46,
    "name": "Test Image 046",
    "description": "SAR satellite image - Oil spill detection test case 46",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637963/oil-spill-test-images/img_0046.jpg",
    "category": "offshore",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 47,
    "name": "Test Image 047",
    "description": "SAR satellite image - Oil spill detection test case 47",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637966/oil-spill-test-images/img_0047.jpg",
    "category": "complex",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 48,
    "name": "Test Image 048",
    "description": "SAR satellite image - Oil spill detection test case 48",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637968/oil-spill-test-images/img_0048.jpg",
    "category": "satellite",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 49,
    "name": "Test Image 049",
    "description": "SAR satellite image - Oil spill detection test case 49",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637969/oil-spill-test-images/img_0049.jpg",
    "category": "coastal",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 50,
    "name": "Test Image 050",
    "description": "SAR satellite image - Oil spill detection test case 50",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637970/oil-spill-test-images/img_0050.jpg",
    "category": "offshore",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 51,
    "name": "Test Image 051",
    "description": "SAR satellite image - Oil spill detection test case 51",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637972/oil-spill-test-images/img_0051.jpg",
    "category": "complex",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 52,
    "name": "Test Image 052",
    "description": "SAR satellite image - Oil spill detection test case 52",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637975/oil-spill-test-images/img_0052.jpg",
    "category": "satellite",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 53,
    "name": "Test Image 053",
    "description": "SAR satellite image - Oil spill detection test case 53",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637976/oil-spill-test-images/img_0053.jpg",
    "category": "coastal",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 54,
    "name": "Test Image 054",
    "description": "SAR satellite image - Oil spill detection test case 54",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637977/oil-spill-test-images/img_0054.jpg",
    "category": "offshore",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 55,
    "name": "Test Image 055",
    "description": "SAR satellite image - Oil spill detection test case 55",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637979/oil-spill-test-images/img_0055.jpg",
    "category": "complex",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 56,
    "name": "Test Image 056",
    "description": "SAR satellite image - Oil spill detection test case 56",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637981/oil-spill-test-images/img_0056.jpg",
    "category": "satellite",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 57,
    "name": "Test Image 057",
    "description": "SAR satellite image - Oil spill detection test case 57",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637983/oil-spill-test-images/img_0057.jpg",
    "category": "coastal",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 58,
    "name": "Test Image 058",
    "description": "SAR satellite image - Oil spill detection test case 58",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637984/oil-spill-test-images/img_0058.jpg",
    "category": "offshore",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 59,
    "name": "Test Image 059",
    "description": "SAR satellite image - Oil spill detection test case 59",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637986/oil-spill-test-images/img_0059.jpg",
    "category": "complex",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 60,
    "name": "Test Image 060",
    "description": "SAR satellite image - Oil spill detection test case 60",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637987/oil-spill-test-images/img_0060.jpg",
    "category": "satellite",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 61,
    "name": "Test Image 061",
    "description": "SAR satellite image - Oil spill detection test case 61",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637989/oil-spill-test-images/img_0061.jpg",
    "category": "coastal",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 62,
    "name": "Test Image 062",
    "description": "SAR satellite image - Oil spill detection test case 62",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637991/oil-spill-test-images/img_0062.jpg",
    "category": "offshore",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 63,
    "name": "Test Image 063",
    "description": "SAR satellite image - Oil spill detection test case 63",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637992/oil-spill-test-images/img_0063.jpg",
    "category": "complex",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 64,
    "name": "Test Image 064",
    "description": "SAR satellite image - Oil spill detection test case 64",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637994/oil-spill-test-images/img_0064.jpg",
    "category": "satellite",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 65,
    "name": "Test Image 065",
    "description": "SAR satellite image - Oil spill detection test case 65",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637995/oil-spill-test-images/img_0065.jpg",
    "category": "coastal",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 66,
    "name": "Test Image 066",
    "description": "SAR satellite image - Oil spill detection test case 66",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637997/oil-spill-test-images/img_0066.jpg",
    "category": "offshore",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 67,
    "name": "Test Image 067",
    "description": "SAR satellite image - Oil spill detection test case 67",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637998/oil-spill-test-images/img_0067.jpg",
    "category": "complex",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 68,
    "name": "Test Image 068",
    "description": "SAR satellite image - Oil spill detection test case 68",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638000/oil-spill-test-images/img_0068.jpg",
    "category": "satellite",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 69,
    "name": "Test Image 069",
    "description": "SAR satellite image - Oil spill detection test case 69",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638002/oil-spill-test-images/img_0069.jpg",
    "category": "coastal",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 70,
    "name": "Test Image 070",
    "description": "SAR satellite image - Oil spill detection test case 70",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638004/oil-spill-test-images/img_0070.jpg",
    "category": "offshore",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 71,
    "name": "Test Image 071",
    "description": "SAR satellite image - Oil spill detection test case 71",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638005/oil-spill-test-images/img_0071.jpg",
    "category": "complex",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 72,
    "name": "Test Image 072",
    "description": "SAR satellite image - Oil spill detection test case 72",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638006/oil-spill-test-images/img_0072.jpg",
    "category": "satellite",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 73,
    "name": "Test Image 073",
    "description": "SAR satellite image - Oil spill detection test case 73",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638008/oil-spill-test-images/img_0073.jpg",
    "category": "coastal",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 74,
    "name": "Test Image 074",
    "description": "SAR satellite image - Oil spill detection test case 74",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638009/oil-spill-test-images/img_0074.jpg",
    "category": "offshore",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 75,
    "name": "Test Image 075",
    "description": "SAR satellite image - Oil spill detection test case 75",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638011/oil-spill-test-images/img_0075.jpg",
    "category": "complex",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 76,
    "name": "Test Image 076",
    "description": "SAR satellite image - Oil spill detection test case 76",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638012/oil-spill-test-images/img_0076.jpg",
    "category": "satellite",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 77,
    "name": "Test Image 077",
    "description": "SAR satellite image - Oil spill detection test case 77",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638013/oil-spill-test-images/img_0077.jpg",
    "category": "coastal",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 78,
    "name": "Test Image 078",
    "description": "SAR satellite image - Oil spill detection test case 78",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638015/oil-spill-test-images/img_0078.jpg",
    "category": "offshore",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 79,
    "name": "Test Image 079",
    "description": "SAR satellite image - Oil spill detection test case 79",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638017/oil-spill-test-images/img_0079.jpg",
    "category": "complex",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 80,
    "name": "Test Image 080",
    "description": "SAR satellite image - Oil spill detection test case 80",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638018/oil-spill-test-images/img_0080.jpg",
    "category": "satellite",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 81,
    "name": "Test Image 081",
    "description": "SAR satellite image - Oil spill detection test case 81",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638020/oil-spill-test-images/img_0081.jpg",
    "category": "coastal",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 82,
    "name": "Test Image 082",
    "description": "SAR satellite image - Oil spill detection test case 82",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638021/oil-spill-test-images/img_0082.jpg",
    "category": "offshore",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 83,
    "name": "Test Image 083",
    "description": "SAR satellite image - Oil spill detection test case 83",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638023/oil-spill-test-images/img_0083.jpg",
    "category": "complex",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 84,
    "name": "Test Image 084",
    "description": "SAR satellite image - Oil spill detection test case 84",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638025/oil-spill-test-images/img_0084.jpg",
    "category": "satellite",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 85,
    "name": "Test Image 085",
    "description": "SAR satellite image - Oil spill detection test case 85",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638026/oil-spill-test-images/img_0085.jpg",
    "category": "coastal",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 86,
    "name": "Test Image 086",
    "description": "SAR satellite image - Oil spill detection test case 86",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638028/oil-spill-test-images/img_0086.jpg",
    "category": "offshore",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 87,
    "name": "Test Image 087",
    "description": "SAR satellite image - Oil spill detection test case 87",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638029/oil-spill-test-images/img_0087.jpg",
    "category": "complex",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 88,
    "name": "Test Image 088",
    "description": "SAR satellite image - Oil spill detection test case 88",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638031/oil-spill-test-images/img_0088.jpg",
    "category": "satellite",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 89,
    "name": "Test Image 089",
    "description": "SAR satellite image - Oil spill detection test case 89",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638033/oil-spill-test-images/img_0089.jpg",
    "category": "coastal",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 90,
    "name": "Test Image 090",
    "description": "SAR satellite image - Oil spill detection test case 90",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638034/oil-spill-test-images/img_0090.jpg",
    "category": "offshore",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 91,
    "name": "Test Image 091",
    "description": "SAR satellite image - Oil spill detection test case 91",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638036/oil-spill-test-images/img_0091.jpg",
    "category": "complex",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 92,
    "name": "Test Image 092",
    "description": "SAR satellite image - Oil spill detection test case 92",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638037/oil-spill-test-images/img_0092.jpg",
    "category": "satellite",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 93,
    "name": "Test Image 093",
    "description": "SAR satellite image - Oil spill detection test case 93",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638038/oil-spill-test-images/img_0093.jpg",
    "category": "coastal",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 94,
    "name": "Test Image 094",
    "description": "SAR satellite image - Oil spill detection test case 94",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638040/oil-spill-test-images/img_0094.jpg",
    "category": "offshore",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 95,
    "name": "Test Image 095",
    "description": "SAR satellite image - Oil spill detection test case 95",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638042/oil-spill-test-images/img_0095.jpg",
    "category": "complex",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 96,
    "name": "Test Image 096",
    "description": "SAR satellite image - Oil spill detection test case 96",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638043/oil-spill-test-images/img_0096.jpg",
    "category": "satellite",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 97,
    "name": "Test Image 097",
    "description": "SAR satellite image - Oil spill detection test case 97",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638044/oil-spill-test-images/img_0097.jpg",
    "category": "coastal",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 98,
    "name": "Test Image 098",
    "description": "SAR satellite image - Oil spill detection test case 98",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638046/oil-spill-test-images/img_0098.jpg",
    "category": "offshore",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 99,
    "name": "Test Image 099",
    "description": "SAR satellite image - Oil spill detection test case 99",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638047/oil-spill-test-images/img_0099.jpg",
    "category": "complex",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 100,
    "name": "Test Image 100",
    "description": "SAR satellite image - Oil spill detection test case 100",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638048/oil-spill-test-images/img_0100.jpg",
    "category": "satellite",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 101,
    "name": "Test Image 101",
    "description": "SAR satellite image - Oil spill detection test case 101",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638050/oil-spill-test-images/img_0101.jpg",
    "category": "coastal",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 102,
    "name": "Test Image 102",
    "description": "SAR satellite image - Oil spill detection test case 102",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638051/oil-spill-test-images/img_0102.jpg",
    "category": "offshore",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 103,
    "name": "Test Image 103",
    "description": "SAR satellite image - Oil spill detection test case 103",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638052/oil-spill-test-images/img_0103.jpg",
    "category": "complex",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 104,
    "name": "Test Image 104",
    "description": "SAR satellite image - Oil spill detection test case 104",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638054/oil-spill-test-images/img_0104.jpg",
    "category": "satellite",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 105,
    "name": "Test Image 105",
    "description": "SAR satellite image - Oil spill detection test case 105",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638056/oil-spill-test-images/img_0105.jpg",
    "category": "coastal",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 106,
    "name": "Test Image 106",
    "description": "SAR satellite image - Oil spill detection test case 106",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638058/oil-spill-test-images/img_0106.jpg",
    "category": "offshore",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 107,
    "name": "Test Image 107",
    "description": "SAR satellite image - Oil spill detection test case 107",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638059/oil-spill-test-images/img_0107.jpg",
    "category": "complex",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 108,
    "name": "Test Image 108",
    "description": "SAR satellite image - Oil spill detection test case 108",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638060/oil-spill-test-images/img_0108.jpg",
    "category": "satellite",
    "difficulty": "easy",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 109,
    "name": "Test Image 109",
    "description": "SAR satellite image - Oil spill detection test case 109",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638062/oil-spill-test-images/img_0109.jpg",
    "category": "coastal",
    "difficulty": "medium",
    "expectedResult": "Oil Spill Analysis"
  },
  {
    "id": 110,
    "name": "Test Image 110",
    "description": "SAR satellite image - Oil spill detection test case 110",
    "url": "https://res.cloudinary.com/darlvqu7v/image/upload/v1751638063/oil-spill-test-images/img_0110.jpg",
    "category": "offshore",
    "difficulty": "hard",
    "expectedResult": "Oil Spill Analysis"
  }
];

// Randomly select images for display
export const getRandomTestImages = (count: number = 20): TestImage[] => {
  const shuffled = [...ALL_TEST_IMAGES].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, count);
};

// Get images by category
export const getImagesByCategory = (category: string): TestImage[] => {
  if (category === 'all') return ALL_TEST_IMAGES;
  return ALL_TEST_IMAGES.filter(img => img.category === category);
};

// Get images by difficulty
export const getImagesByDifficulty = (difficulty: string): TestImage[] => {
  if (difficulty === 'all') return ALL_TEST_IMAGES;
  return ALL_TEST_IMAGES.filter(img => img.difficulty === difficulty);
};

// Default test images (random 20)
export const TEST_IMAGES: TestImage[] = getRandomTestImages(20);

// Category colors for UI
export const DIFFICULTY_COLORS = {
  easy: "bg-green-100 text-green-800 border-green-200",
  medium: "bg-yellow-100 text-yellow-800 border-yellow-200",
  hard: "bg-red-100 text-red-800 border-red-200"
};

// Category icons mapping
export const CATEGORY_ICONS = {
  "satellite": "🛰️",
  "coastal": "🏖️",
  "offshore": "🌊",
  "complex": "🌀",
  "test-data": "📊"
};

// Available categories and difficulties
export const CATEGORIES = ['all', 'satellite', 'coastal', 'offshore', 'complex'];
export const DIFFICULTIES = ['all', 'easy', 'medium', 'hard'];
